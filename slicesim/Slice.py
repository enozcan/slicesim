import simpy


class Slice:
    def __init__(self, name, ratio,
                 connected_users, user_share, delay_tolerance, qos_class,
                 bandwidth_guaranteed, bandwidth_max, init_capacity,
                 usage_pattern, env, index):
        self.name = name
        self.connected_users = connected_users
        self.user_share = user_share
        self.delay_tolerance = delay_tolerance
        self.qos_class = qos_class
        self.ratio = ratio
        self.bandwidth_guaranteed = bandwidth_guaranteed
        self.bandwidth_max = bandwidth_max
        self.init_capacity = init_capacity
        self.capacity = simpy.Container(env, init=init_capacity, capacity=init_capacity)
        self.usage_pattern = usage_pattern
        self.index = index

    def get_consumable_share(self):
        if self.connected_users <= 0:
            return min(self.init_capacity, self.bandwidth_max)
        else:
            return min(self.init_capacity / self.connected_users, self.bandwidth_max)

    def is_available(self):
        bandwidth_next = min(self.init_capacity / (self.connected_users + 1), self.bandwidth_max)
        if bandwidth_next < self.bandwidth_guaranteed:
            return False
        return True

    def print_stats(self):
        real_cap = min(self.init_capacity, self.bandwidth_max)
        print("init cap:", self.init_capacity, "bandwidth_max", self.bandwidth_max)
        bandwidth_next = real_cap / (self.connected_users + 1)
        print("bandwidth_next", bandwidth_next, "bandwidth_guaranteed", self.bandwidth_guaranteed)
        print("connected users:", self.connected_users)

    def get_load(self):
        return 1.0 - (self.capacity.level / self.capacity.capacity)

    def __str__(self):
        return f'{self.name:<10} init={self.init_capacity:<5} cap={self.capacity.level:<5}' \
               f' diff={(self.init_capacity - self.capacity.level):<5} '
