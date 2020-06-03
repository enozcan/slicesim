class BaseStation:
    def __init__(self, pk, coverage, capacity_bandwidth, slices=None):
        self.pk = pk
        self.coverage = coverage
        self.capacity_bandwidth = capacity_bandwidth
        self.slices = slices

    def __str__(self):
        return f'BS_{self.pk:<2}\t cov:{self.coverage}\t with cap {self.capacity_bandwidth:<5}'

    def has_slice(self, slice_name):
        return any(s for s in self.slices if s.name == slice_name)

    def is_neighbour(self, other):
        dist_sq = (self.coverage.center[0] - other.coverage.center[0]) ** 2 \
                  + (self.coverage.center[1] - other.coverage.center[1]) ** 2
        rad_sum_sq = (self.coverage.radius + other.coverage.radius) ** 2
        return dist_sq < rad_sum_sq


