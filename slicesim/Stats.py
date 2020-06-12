import numpy as np
from collections import defaultdict


class Stats:
    def __init__(self, env, base_stations, clients, area):
        self.env = env
        self.base_stations = base_stations
        self.clients = clients
        self.area = area
        # self.graph = graph

        # Stats
        self.total_connected_users_ratio = []
        self.total_used_bw = []
        self.avg_slice_load_ratio = []
        self.avg_slice_client_count_ratio = []
        self.coverage_ratio = []
        self.connect_attempt = []

        # Block count -> the client requests for a resource but
        # the resource is not allocated due to unavailable
        self.block_count_ratio = []

        self.handover_count_ratio = []
        self.drop_count_ratio = []

        self.load_stats = {}
        for bs in self.base_stations:
            self.load_stats[bs.pk] = {}
            for sl in bs.slices:
                self.load_stats[bs.pk][sl.name] = []

    def get_stats(self):
        return (
            self.total_connected_users_ratio,
            self.total_used_bw,
            self.avg_slice_load_ratio,
            self.avg_slice_client_count_ratio,
            self.coverage_ratio,
            self.block_count_ratio,
            self.handover_count_ratio,
            self.drop_count_ratio,
        )

    def collect(self):
        yield self.env.timeout(0.25)
        self.connect_attempt.append(0)
        self.block_count_ratio.append(0)
        self.handover_count_ratio.append(0)
        self.drop_count_ratio.append(0)

        while True:
            self.block_count_ratio[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1
            self.handover_count_ratio[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1

            self.drop_count_ratio[-1] /= (self.connect_attempt[-1] + self.drop_count_ratio[-1]) if \
                self.connect_attempt[-1] != 0 else 1

            self.total_connected_users_ratio.append(self.get_total_connected_users_ratio())
            self.total_used_bw.append(self.get_total_used_bw())
            self.avg_slice_load_ratio.append(self.get_avg_slice_load_ratio())
            self.avg_slice_client_count_ratio.append(self.get_avg_slice_client_count())
            self.coverage_ratio.append(self.get_coverage_ratio())

            self.connect_attempt.append(0)
            self.block_count_ratio.append(0)
            self.handover_count_ratio.append(0)
            self.drop_count_ratio.append(0)

            yield self.env.timeout(1)

    def get_total_connected_users_ratio(self):
        t, cc = 0, 0
        for c in self.clients:
            if self.is_client_in_coverage(c):
                t += c.connected
                cc += 1
        # for bs in self.base_stations:
        #     for sl in bs.slices:
        #         t += sl.connected_users
        return t / cc if cc != 0 else 0

    def get_total_used_bw(self):
        t = 0
        for bs in self.base_stations:
            for sl in bs.slices:
                t += 1e-9 * (sl.capacity.capacity - sl.capacity.level)
        return t

    def get_avg_slice_load_ratio(self):
        t, c = 0, 0
        for bs in self.base_stations:
            for sl in bs.slices:
                c += sl.capacity.capacity
                t += sl.capacity.capacity - sl.capacity.level
                # c += 1
                # t += (sl.capacity.capacity - sl.capacity.level) / sl.capacity.capacity
                self.load_stats[bs.pk][sl.name].append(sl.get_load())
        return t / c if c != 0 else 0

    def get_avg_slice_client_count(self):
        t, c = 0, 0
        for bs in self.base_stations:
            for sl in bs.slices:
                c += 1
                t += sl.connected_users
        return t / c if c != 0 else 0

    def get_coverage_ratio(self):
        t, cc = 0, 0
        for c in self.clients:
            if self.is_client_in_coverage(c):
                cc += 1
                if c.base_station is not None and c.base_station.coverage.is_in_coverage(c.x, c.y):
                    t += 1
        return t / cc if cc != 0 else 0

    def incr_connect_attempt(self, client):
        if self.is_client_in_coverage(client):
            self.connect_attempt[-1] += 1

    def incr_drop_count(self, client):
        if self.is_client_in_coverage(client):
            self.drop_count_ratio[-1] += 1

    def incr_block_count(self, client):
        if self.is_client_in_coverage(client):
            self.block_count_ratio[-1] += 1

    def incr_handover_count(self, client):
        if self.is_client_in_coverage(client):
            self.handover_count_ratio[-1] += 1

    def is_client_in_coverage(self, client):
        xs, ys = self.area
        return True if xs[0] <= client.x <= xs[1] and ys[0] <= client.y <= ys[1] else False

    def get_general_stats(self):
        return {'total_connected_users_ratio': self.total_connected_users_ratio,
                'total_used_bw': self.total_used_bw,
                'avg_slice_load_ratio': self.avg_slice_load_ratio,
                'avg_slice_client_count_ratio': self.avg_slice_client_count_ratio,
                'coverage_ratio': self.coverage_ratio,
                'block_count_ratio': self.block_count_ratio,
                'handover_count_ratio': self.handover_count_ratio,
                'drop_count_ratio': self.drop_count_ratio}

    def get_per_slice_stats(self):
        res = {}
        slices = defaultdict(lambda: [])
        load_per_time = defaultdict(lambda: np.asarray([]))
        for bs, slice_meta in self.load_stats.items():
            for slice_name, load_list in slice_meta.items():
                slices[slice_name].append(np.mean(load_list))
                if len(load_per_time[slice_name]) is 0:
                    load_per_time[slice_name] = load_list
                else:
                    load_per_time[slice_name] = (np.asarray(load_per_time[slice_name]) + np.asarray(load_list)) / 2
        for k, v in slices.items():
            res[k] = (np.mean(v), np.std(v), v)
            # series = [f'{elem:.8f}' for elem in v]
            # print(f'[{k}]:\tMean: {np.mean(v):.4f}, Dev: {np.std(v):.4f}, Values: {series}')
        return res, load_per_time

    def print_detailed_slice_load_stats(self):
        print('-' * 20, "Station Load Stats", '-' * 20)
        for bs, slice_meta in self.load_stats.items():
            print('-' * 10, "BS:", bs, '-' * 10)
            for slice_name, load_list in slice_meta.items():
                res = [f'{elem:.2f}' for elem in load_list]
                print(f'[{slice_name}]\t Mean: {np.mean(load_list):.4f}, Dev: {np.std(load_list):.4f}, Values: {"res"}')
        print('-' * 60)

