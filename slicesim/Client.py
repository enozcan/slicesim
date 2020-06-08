import random
import numpy as np
import os
from .utils import distance, KDTree, LoadBalanceType

PER_SLICE_THRESHOLD = 0.6
HAND_OVER_LOAD_MARGIN = 0.05

class Client:
    def __init__(self, pk, env, x, y, mobility_pattern,
                 usage_freq,
                 subscribed_slice_indices, stat_collector,
                 lb_handover_type, base_station=None):
        self.pk = pk
        self.env = env
        self.x = x
        self.y = y
        self.mobility_pattern = mobility_pattern
        self.usage_freq = usage_freq
        self.base_station = base_station
        self.stat_collector = stat_collector
        self.subscribed_slice_indices = subscribed_slice_indices
        self.usage_remaining = {}
        self.last_usage = {}
        for index in self.subscribed_slice_indices:
            self.usage_remaining[index] = 0
            self.last_usage[index] = 0
        self.closest_base_stations = []
        self.connected = False

        # Stats
        self.total_connected_time = 0
        self.total_unconnected_time = 0
        self.total_request_count = 0
        self.total_consume_time = 0
        self.total_usage = 0

        self.action = env.process(self.iter())
        # print(self.usage_freq)

        self.suppress_log = True if os.environ["SLICE_SIM_LOG_STAT_ONLY"] is "1" else False
        self.lb_handover_type = lb_handover_type

    def get_slice_balance_load(self, station):
        """
        Returns the load level of a given station considering only the slices used by this client.
        The returned value might differ according to the handover logic.
        For instance, in LoadBalanceType.max, the maximum load of these slices are returned.
        :param station: Base station of load to be calculated
        :return: Load value according to load balance logic
        """
        if self.lb_handover_type is LoadBalanceType.disabled:
            return -1  # ignored
        elif self.lb_handover_type is LoadBalanceType.max:
            return max([s.get_load() for s in np.asarray(station.slices)[self.subscribed_slice_indices]])
        elif self.lb_handover_type is LoadBalanceType.mean:
            return np.mean([s.get_load() for s in np.asarray(station.slices)[self.subscribed_slice_indices]])
        else:
            raise NotImplementedError

    def get_current_lb_load(self):
        """
        Calculates the currently connected base station's slice load considering only the slices
        used by this client.
        :return: Load value according to load balance logic
        """
        current_slice_loads = [s.get_load() for s in self.get_slices()] if self.base_station is not None else []
        if self.lb_handover_type is LoadBalanceType.disabled:
            return -1  # ignored
        elif self.lb_handover_type is LoadBalanceType.max:
            return max(current_slice_loads) if len(current_slice_loads) is not 0 else -1
        elif self.lb_handover_type is LoadBalanceType.mean:
            return np.mean(current_slice_loads) if len(current_slice_loads) is not 0 else -1
        else:
            raise NotImplementedError

    def should_skip_lb_handover(self, current_load, candidate_load):
        """
        Decides if this client should handover in the next iteration or not.
        :param current_load: result of self.get_current_lb_load
        :param candidate_load: result of get_slice_balance_load for a candidate base station
        :return: True if handover should be performed, False otherwise
        """
        if self.lb_handover_type is LoadBalanceType.disabled:
            return True  # do not attempt to make load balance. Skip directly
        elif self.lb_handover_type is LoadBalanceType.max or LoadBalanceType.mean:
            return current_load < PER_SLICE_THRESHOLD or \
                   candidate_load > (current_load - HAND_OVER_LOAD_MARGIN)
        else:
            raise NotImplementedError

    def get_next_base_station(self):
        """
        Applies handover logic and chooses a base station among available ones including the current one
        :return: Base station to be connected inn the next time unit. Might be None or the same as the current one.
        """
        in_coverage = self.base_station is not None and self.base_station.coverage.is_in_coverage(self.x, self.y)
        current_load = self.get_current_lb_load()

        st = self.get_candidate_base_stations(exclude=[self.base_station.pk] if self.base_station is not None else [])
        st.sort(key=lambda x: self.get_slice_balance_load(x[1]))  # TODO: Pass lambda as param for distinct mechanisms
        candidate_load = self.get_slice_balance_load(st[0][1]) if len(st) > 0 else 1

        if in_coverage and self.should_skip_lb_handover(current_load, candidate_load):
            return self.base_station

        self.log(f'[{int(self.env.now)}] Client_{self.pk} old load was {current_load} at '
                 f'BS:{self.base_station.pk if self.base_station is not None else None}, '
                 f'new load is {candidate_load} at BS:{st[0][1].pk if len(st) > 0 else None}')
        return st[0][1] if len(st) > 0 else None

    def assign_optimal_base_station(self):
        """
        Assigns the optimal base station after handover logic is applied.
        :return: True if handover is performed, False otherwise
        """

        if KDTree.last_run_time is not int(self.env.now):
            KDTree.run(self.stat_collector.clients, self.stat_collector.base_stations, int(self.env.now),
                       assign=False, logging=(not self.suppress_log))

        next_bs = self.get_next_base_station()
        if self.base_station is next_bs:
            self.log(f'[{int(self.env.now)}] Client_{self.pk} continues to be assigned to {self.base_station}')
            return False

        if self.base_station is None:
            self.base_station = next_bs
            self.log(f'[{int(self.env.now)}] Client_{self.pk} freshly assigned to {self.base_station}')
            return False

        if self.connected:
            self.log(f'[{int(self.env.now)}] Client_{self.pk} disconnecting from {self.base_station.pk}')
            self.disconnect()

        if next_bs is None:
            self.log(f'[{int(self.env.now)}] Client_{self.pk} could not assigned to any base station')
            self.stat_collector.incr_drop_count(self)
            self.base_station = next_bs
            return False

        # handover happens here.
        self.log(f'[{int(self.env.now)}] Client_{self.pk} assigned to {next_bs} after handover.')
        self.base_station = next_bs
        self.stat_collector.incr_handover_count(self)
        return True

    def is_all_remaining_usages_zero(self):
        for _, v in self.usage_remaining.items():
            if v is not 0:
                return False
        return True

    def is_all_last_usages_zero(self):
        for _, v in self.last_usage.items():
            if v is not 0:
                return False
        return True

    def iter(self):
        """
        There are four steps in a cycle:
            1- .00: Lock
            2- .25: Stats
            3- .50: Release
            4- .75: Move
        """

        # .00: Lock
        handover_performed = self.assign_optimal_base_station()

        if self.base_station is not None:
            if not self.is_all_remaining_usages_zero():
                self.generate_usage()
                if self.connected:
                    self.start_consume()
                else:
                    self.connect(handover_performed)
            else:
                if self.connected:
                    self.disconnect()
                else:
                    if self.generate_usage():
                        self.connect()

        yield self.env.timeout(0.25)

        # .25: Stats

        yield self.env.timeout(0.25)

        # .50: Release
        # Base station check skipped as it's already implied by self.connected
        if self.connected and not self.is_all_last_usages_zero():
            self.release_consume()
            if self.is_all_remaining_usages_zero():
                self.disconnect()

        yield self.env.timeout(0.25)

        # .75: Move
        # Move the client
        x, y = self.mobility_pattern.generate_movement()
        self.x += x
        self.y += y
        """
        if self.base_station is not None:
            if not self.base_station.coverage.is_in_coverage(self.x, self.y):
                self.disconnect()
                self.assign_closest_base_station(exclude=[self.base_station.pk])
            elif self.should_handover(prev_load):
                self.handover()
        else:
            self.assign_closest_base_station()
        """

        yield self.env.timeout(0.25)

        yield self.env.process(self.iter())

    def get_slices(self):
        if self.base_station is None:
            return None
        return np.asarray(self.base_station.slices)[self.subscribed_slice_indices]

    def generate_usage(self):
        generated = False
        if self.get_slices() is None:
            return generated
        for slice_idx, remain in self.usage_remaining.items():
            if remain is 0 and self.usage_freq < random.random():
                sl = self.base_station.slices[slice_idx]
                self.usage_remaining[slice_idx] = sl.usage_pattern.generate()
                self.total_request_count += 1
                self.log(
                    f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] requests {self.usage_remaining[slice_idx]}'
                    f' usage from slice: {sl}')
                generated = True
        return generated

    def is_bs_available(self):
        for sl in self.get_slices():
            if self.usage_remaining[sl.index] > 0 and not sl.is_available():
                self.log(f'[{int(self.env.now)}] Client_{self.pk} is blocked at bs {self.base_station} for slice {sl.name} '
                      f'and its load={sl.get_load()}, its availability={sl.is_available()}')
                sl.print_stats()
                return False
        return True

    def connect(self, handover_performed=False):
        if self.connected:
            return
        slices = self.get_slices()
        # increment connect attempt
        self.stat_collector.incr_connect_attempt(self)
        if self.is_bs_available():
            for sl in slices:
                sl.connected_users += 1
            self.connected = True
            self.log(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] connected to slices={[s.name for s in slices]}'
                f' @ {self.base_station}')
            return True
        else:
            """ from the old version of SliceSim:
            
            self.assign_closest_base_station(exclude=[self.base_station.pk])
            if self.base_station is not None and self.get_slice().is_available():
                # handover
                self.stat_collector.incr_handover_count(self)
            elif self.base_station is not None:
                # block
                self.stat_collector.incr_block_count(self)
            else:
                pass  # uncovered
                
            """
            if handover_performed:
                self.stat_collector.incr_drop_count(self)
            else:
                self.stat_collector.incr_block_count(self)
            self.log(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] connection refused to '
                f'slices={[s.name for s in slices]} @ {self.base_station}')
            return False

    def disconnect(self):
        slices = self.get_slices()
        if not self.connected:
            self.log(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] is already disconnected from '
                f'slices={[s.name for s in slices]} @ {self.base_station}')
        else:
            for sl in slices:
                sl.connected_users -= 1
            self.connected = False
            self.log(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] disconnected from'
                f' slices={[s.name for s in slices]} @ {self.base_station}')
        return not self.connected

    def start_consume(self):
        slices = self.get_slices()
        for s in slices:
            amount = min(s.get_consumable_share(), self.usage_remaining[s.index])
            # Allocate resource and consume ongoing usage with given bandwidth
            if amount <= 0:
                self.last_usage[s.index] = 0
                continue
            s.capacity.get(amount)
            self.log(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] gets {amount} usage from slice: {s}.')
            self.last_usage[s.index] = amount

    def release_consume(self):
        slices = self.get_slices()
        for s in slices:
            # Put the resource back
            last_usage = self.last_usage[s.index]
            if last_usage > 0:  # note: s.capacity.put cannot take 0
                s.capacity.put(last_usage)
                self.log(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] puts back {last_usage} usage.')
                self.total_consume_time += 1
                self.total_usage += last_usage
                self.usage_remaining[s.index] -= last_usage
                self.last_usage[s.index] = 0

    def get_candidate_base_stations(self, exclude=None):
        updated_list = []
        for d, b in self.closest_base_stations:
            if exclude is not None and b.pk in exclude:
                continue
            d = distance((self.x, self.y), (b.coverage.center[0], b.coverage.center[1]))
            updated_list.append((d, b))
        # updated_list.sort(key=operator.itemgetter(0))  already sorted according to
        filtered_list = [x for x in updated_list if x[0] <= x[1].coverage.radius]
        return filtered_list

    def __str__(self):
        return f'Client_{self.pk} [{self.x:<5}, {self.y:>5}] connected to: slices={[s.name for s in self.get_slices()]} ' \
               f'@ {self.base_station}\t with mobility pattern of {self.mobility_pattern}'

    def log(self, message):
        if self.suppress_log:
            return
        print(message)
