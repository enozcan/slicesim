import operator
import random
import numpy as np
from .utils import distance, KDTree

HAND_OFF_THRESHOLD = 0.1


class Client:
    def __init__(self, pk, env, x, y, mobility_pattern,
                 usage_freq,
                 subscribed_slice_index, stat_collector,
                 base_station=None):
        self.pk = pk
        self.env = env
        self.x = x
        self.y = y
        self.mobility_pattern = mobility_pattern
        self.usage_freq = usage_freq
        self.base_station = base_station
        self.stat_collector = stat_collector
        self.subscribed_slice_index = subscribed_slice_index
        self.usage_remaining = 0
        self.last_usage = 0
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

    def iter(self):
        """
        There are four steps in a cycle:
            1- .00: Lock
            2- .25: Stats
            3- .50: Release
            4- .75: Move
        """

        # .00: Lock
        if self.base_station is not None:
            if self.usage_remaining > 0:
                if self.connected:
                    self.start_consume()
                else:
                    self.connect()
            else:
                if self.connected:
                    self.disconnect()
                else:
                    self.generate_usage_and_connect()

        yield self.env.timeout(0.25)

        # .25: Stats

        if self.get_slice():
            prev_load = self.get_slice().get_load()
        else:
            prev_load = -1

        yield self.env.timeout(0.25)

        # .50: Release
        # Base station check skipped as it's already implied by self.connected
        if self.connected and self.last_usage > 0:
            self.release_consume()
            if self.usage_remaining <= 0:
                self.disconnect()

        yield self.env.timeout(0.25)

        # .75: Move
        # Move the client
        x, y = self.mobility_pattern.generate_movement()
        self.x += x
        self.y += y

        if self.base_station is not None:
            if not self.base_station.coverage.is_in_coverage(self.x, self.y):
                self.disconnect()
                self.assign_closest_base_station(exclude=[self.base_station.pk])
            elif self.should_handover(prev_load):
                self.handover()
        else:
            self.assign_closest_base_station()

        yield self.env.timeout(0.25)

        yield self.env.process(self.iter())

    def should_handover(self, load):
        st = self.get_closest_base_stations(exclude=[self.base_station.pk])
        st = [x for x in st if x[0] <= x[1].coverage.radius]
        # st contains the available stations in the range now.
        # if it's empty, then handover is not available for this client.
        print("LOAD:", load, "for BS:", self.base_station.pk, "for Slice:", self.get_slice().name)
        if load >= HAND_OFF_THRESHOLD and len(st) is not 0:
            if np.random.random(1)[0] > ((1.0 * load - HAND_OFF_THRESHOLD) / load):
                return True
            else:
                return False

    def handover(self):
        old_pk = self.base_station.pk
        self.disconnect()
        self.assign_closest_base_station(exclude=[self.base_station.pk])
        print(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] handed off from {old_pk} to {self.base_station.pk if self.base_station is not None else None}')

    def get_slice(self):
        if self.base_station is None:
            return None
        return self.base_station.slices[self.subscribed_slice_index]

    def generate_usage_and_connect(self):
        if self.usage_freq < random.random() and self.get_slice() is not None:
            # Generate a new usage
            self.usage_remaining = self.get_slice().usage_pattern.generate()
            self.total_request_count += 1
            self.connect()
            print(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] requests {self.usage_remaining} usage.')

    def connect(self):
        s = self.get_slice()
        if self.connected:
            return
        # increment connect attempt
        self.stat_collector.incr_connect_attempt(self)
        if s.is_available():
            s.connected_users += 1
            self.connected = True
            print(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] connected to slice={self.get_slice()}'
                f' @ {self.base_station}')
            return True
        else:
            self.assign_closest_base_station(exclude=[self.base_station.pk])
            if self.base_station is not None and self.get_slice().is_available():
                # handover
                self.stat_collector.incr_handover_count(self)
            elif self.base_station is not None:
                # block
                self.stat_collector.incr_block_count(self)
            else:
                pass  # uncovered
            print(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] connection refused to '
                f'slice={self.get_slice()} @ {self.base_station}')
            return False

    def disconnect(self):
        if not self.connected:
            print(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] is already disconnected from '
                f'slice={self.get_slice()} @ {self.base_station}')
        else:
            slice = self.get_slice()
            slice.connected_users -= 1
            self.connected = False
            print(
                f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] disconnected from'
                f' slice={self.get_slice()} @ {self.base_station}')
        return not self.connected

    def start_consume(self):
        s = self.get_slice()
        amount = min(s.get_consumable_share(), self.usage_remaining)
        # Allocate resource and consume ongoing usage with given bandwidth
        s.capacity.get(amount)
        print("Load of the slice:", s.get_load())
        print(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] gets {amount} usage.')
        self.last_usage = amount

    def release_consume(self):
        s = self.get_slice()
        # Put the resource back
        if self.last_usage > 0:  # note: s.capacity.put cannot take 0
            s.capacity.put(self.last_usage)
            print(f'[{int(self.env.now)}] Client_{self.pk} [{self.x}, {self.y}] puts back {self.last_usage} usage.')
            self.total_consume_time += 1
            self.total_usage += self.last_usage
            self.usage_remaining -= self.last_usage
            self.last_usage = 0

    # Check closest base_stations of a client and assign the closest non-excluded available base_station to the client.
    def assign_closest_base_station(self, exclude=None):
        stations = self.get_closest_base_stations(exclude)
        for d, b in stations:
            if d <= b.coverage.radius:
                self.base_station = b
                print(f'[{int(self.env.now)}] Client_{self.pk} freshly assigned to {self.base_station}')
                return
        if KDTree.last_run_time is not int(self.env.now):
            KDTree.run(self.stat_collector.clients, self.stat_collector.base_stations, int(self.env.now), assign=False)
        self.base_station = None

    def get_closest_base_stations(self, exclude=None):
        updated_list = []
        for d, b in self.closest_base_stations:
            if exclude is not None and b.pk in exclude:
                continue
            d = distance((self.x, self.y), (b.coverage.center[0], b.coverage.center[1]))
            updated_list.append((d, b))
        updated_list.sort(key=operator.itemgetter(0))
        return updated_list

    def __str__(self):
        return f'Client_{self.pk} [{self.x:<5}, {self.y:>5}] connected to: slice={self.get_slice()} ' \
               f'@ {self.base_station}\t with mobility pattern of {self.mobility_pattern}'
