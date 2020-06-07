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
        self.avg_slice_client_count = []
        self.coverage_ratio = []
        self.connect_attempt = []

        # Block count -> the client requests for a resource but
        # the resource is not allocated due to unavailable
        self.block_count = []

        self.handover_count = []
        self.drop_count = []

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
            self.avg_slice_client_count,
            self.coverage_ratio,
            self.block_count,
            self.handover_count,
        )

    def collect(self):
        yield self.env.timeout(0.25)
        self.connect_attempt.append(0)
        self.block_count.append(0)
        self.handover_count.append(0)
        self.drop_count.append(0)

        while True:
            self.block_count[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1
            self.handover_count[-1] /= self.connect_attempt[-1] if self.connect_attempt[-1] != 0 else 1

            self.drop_count[-1] /= (self.connect_attempt[-1] + self.drop_count[-1]) if \
                self.connect_attempt[-1] != 0 else 1

            self.total_connected_users_ratio.append(self.get_total_connected_users_ratio())
            self.total_used_bw.append(self.get_total_used_bw())
            self.avg_slice_load_ratio.append(self.get_avg_slice_load_ratio())
            self.avg_slice_client_count.append(self.get_avg_slice_client_count())
            self.coverage_ratio.append(self.get_coverage_ratio())

            self.connect_attempt.append(0)
            self.block_count.append(0)
            self.handover_count.append(0)
            self.drop_count.append(0)

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
            self.drop_count[-1] += 1

    def incr_block_count(self, client):
        if self.is_client_in_coverage(client):
            self.block_count[-1] += 1

    def incr_handover_count(self, client):
        if self.is_client_in_coverage(client):
            self.handover_count[-1] += 1

    def is_client_in_coverage(self, client):
        xs, ys = self.area
        return True if xs[0] <= client.x <= xs[1] and ys[0] <= client.y <= ys[1] else False

    def print_general_stats(self):
        print('-' * 20, "General Stats", '-' * 20)
        r = lambda stats: [f'{elem:.4f}' for elem in stats]
        p = lambda header, stats: print(f'[{header}] Mean: {np.mean(stats):.4f}, Dev: {np.std(stats):.4f}, '
                                        f'Series: {r(stats)}')
        p("Total Connected Users\t", self.total_connected_users_ratio)
        p("Total Bandwidth Used (Gbps)\t", self.total_used_bw)
        p("Avg Slice Load Ratio\t", self.avg_slice_load_ratio)
        p("Avg Slice Client Count\t", self.avg_slice_client_count)
        p("Client Coverage Ratios\t", self.coverage_ratio)
        p("Blocked Clients Count\t", self.block_count)
        p("Client Handover Count\t", self.handover_count)
        p("Drop Count Rate\t", self.drop_count)
        print('-' * 60)

    def print_detailed_slice_load_stats(self):
        self.print_per_slice_stats()
        print('-' * 20, "Station Load Stats", '-' * 20)
        for bs, slice_meta in self.load_stats.items():
            print('-' * 10, "BS:", bs, '-' * 10)
            for slice_name, load_list in slice_meta.items():
                res = [f'{elem:.2f}' for elem in load_list]
                print(f'[{slice_name}]\t Mean: {np.mean(load_list):.4f}, Dev: {np.std(load_list):.4f}, Values: {"res"}')
        print('-' * 60)

    def print_per_slice_stats(self):
        print('-' * 20, "Slice Load Stats", '-' * 20)
        slices = defaultdict(lambda: [])
        for bs, slice_meta in self.load_stats.items():
            for slice_name, load_list in slice_meta.items():
                slices[slice_name].append(np.mean(load_list))
        for k, v in slices.items():
            res = [f'{elem:.8f}' for elem in v]
            print(f'[{k}]:\tMean: {np.mean(v):.4f}, Dev: {np.std(v):.4f}, Values: {"res"}')
        print('-' * 60)
