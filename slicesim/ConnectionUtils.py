from itertools import combinations


def get_connection_matrices(slices, base_stations):
    """
    Given list of slice names and base station objects,
    returns 2d connection matrix for each slice.
    i.e.
    if result['slice_1'][<bs1>][<bs2>] == 1 then
    bs1 and bs2 share a common area and both have slice_1.

    :param slices:          List of slice names
    :param base_stations:   List of base station objects
    :return:                2d connection matrices for each slice.
    """
    result = {}
    for s in slices:
        result[s] = [[0 for _ in range(len(base_stations))] for _ in (range(len(base_stations)))]
        for comb in list(combinations(base_stations, 2)):
            if comb[0].is_neighbour(comb[1]) \
                    and comb[0].has_slice(s) \
                    and comb[1].has_slice(s):
                result[s][comb[0].pk][comb[1].pk] = 1
                result[s][comb[1].pk][comb[0].pk] = 1
    return result
