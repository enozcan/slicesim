import os
import random
import sys
import numpy as np

import simpy
import yaml

from .BaseStation import BaseStation
from .Client import Client
from .Coverage import Coverage
from .Distributor import Distributor
from .Graph import Graph
from .Slice import Slice
from .Stats import Stats

from .utils import KDTree


def get_dist(d):
    return {
        'randrange': random.randrange,  # start, stop, step
        'randint': random.randint,  # a, b
        'random': random.random,
        'uniform': random,  # a, b
        'triangular': random.triangular,  # low, high, mode
        'beta': random.betavariate,  # alpha, beta
        'expo': random.expovariate,  # lambda
        'gamma': random.gammavariate,  # alpha, beta
        'gauss': random.gauss,  # mu, sigma
        'lognorm': random.lognormvariate,  # mu, sigma
        'normal': random.normalvariate,  # mu, sigma
        'vonmises': random.vonmisesvariate,  # mu, kappa
        'pareto': random.paretovariate,  # alpha
        'weibull': random.weibullvariate  # alpha, beta
    }.get(d)


def get_random_mobility_pattern(vals, mobility_patterns):

    i = 0
    r = random.random()

    while vals[i] < r:
        i += 1

    return mobility_patterns[i]


def get_random_slice_indices(vals):
    subscribed_slices_count = np.random.randint(3, size=1)[0] + 1
    result = np.random.choice(len(vals), subscribed_slices_count, replace=False, p=vals)
    return result


if len(sys.argv) != 3:
    print('Please type an input file.')
    print('python -m slicesim <input-file>')
    exit(1)

# Read YAML file
CONF_FILENAME = os.path.join(os.path.dirname(__file__), sys.argv[2])
try:
    with open(CONF_FILENAME, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
except FileNotFoundError:
    print('File Not Found:', CONF_FILENAME)
    exit(0)

random.seed()
env = simpy.Environment()

SETTINGS = data['settings']
SLICES_INFO = data['slices']
NUM_CLIENTS = SETTINGS['num_clients']
MOBILITY_PATTERNS = data['mobility_patterns']
BASE_STATIONS = data['base_stations']
CLIENTS = data['clients']
LB_HANDOVER = SETTINGS['load_balance_handover']

if SETTINGS['logging']:
    sys.stdout = open(SETTINGS['log_file'], 'wt')
else:
    sys.stdout = open(os.devnull, 'w')

os.environ["SLICE_SIM_LOG_STAT_ONLY"] = "1" if SETTINGS['log_stat_only'] else "0"

collected, slice_weights = 0, []
for __, s in SLICES_INFO.items():
    # collected += s['client_weight']
    slice_weights.append(s['client_weight'])

collected, mb_weights = 0, []
for __, mb in MOBILITY_PATTERNS.items():
    collected += mb['client_weight']
    mb_weights.append(collected)

mobility_patterns = []
for name, mb in MOBILITY_PATTERNS.items():
    mobility_pattern = Distributor(name, get_dist(mb['distribution']), *mb['params'])
    mobility_patterns.append(mobility_pattern)

usage_patterns = {}
for name, s in SLICES_INFO.items():
    usage_patterns[name] = Distributor(name, get_dist(s['usage_pattern']['distribution']),
                                       *s['usage_pattern']['params'])

print('-' * 20, "Base Stations", '-' * 20)
base_stations = []
i = 0
for b in BASE_STATIONS:
    slices = []
    ratios = b['ratios']
    capacity = b['capacity_bandwidth']
    slice_idx = 0
    for name, s in SLICES_INFO.items():
        s_cap = capacity * ratios[name]
        # TODO remove bandwidth max
        s = Slice(name, ratios[name], 0, s['client_weight'],
                  s['delay_tolerance'],
                  s['qos_class'], s['bandwidth_guaranteed'],
                  s['bandwidth_max'], s_cap, usage_patterns[name], env, slice_idx)
        slices.append(s)
        slice_idx += 1
    base_station = BaseStation(i, Coverage((b['x'], b['y']), b['coverage']), capacity, slices)
    base_stations.append(base_station)
    print(base_station)
    i += 1
print('-' * 60)

ufp = CLIENTS['usage_frequency']
usage_freq_pattern = Distributor(f'ufp', get_dist(ufp['distribution']), *ufp['params'],
                                 divide_scale=ufp['divide_scale'])

x_vals = SETTINGS['statistics_params']['x']
y_vals = SETTINGS['statistics_params']['y']
stats = Stats(env, base_stations, None, ((x_vals['min'], x_vals['max']), (y_vals['min'], y_vals['max'])))

clients = []

for i in range(NUM_CLIENTS):
    loc_x = CLIENTS['location']['x']
    loc_y = CLIENTS['location']['y']
    location_x = get_dist(loc_x['distribution'])(*loc_x['params'])
    location_y = get_dist(loc_y['distribution'])(*loc_y['params'])

    mobility_pattern = get_random_mobility_pattern(mb_weights, mobility_patterns)
    connected_slice_indices = get_random_slice_indices(slice_weights)
    c = Client(i, env, location_x, location_y,
               mobility_pattern, usage_freq_pattern.generate_scaled(), connected_slice_indices, stats,
               lb_handover=LB_HANDOVER)
    clients.append(c)

KDTree.limit = SETTINGS['limit_closest_base_stations']
KDTree.run(clients, base_stations, 0, logging=False if os.environ["SLICE_SIM_LOG_STAT_ONLY"] is "1" else True)

stats.clients = clients
env.process(stats.collect())

env.run(until=int(SETTINGS['simulation_time']))

# TODO: Some stats of clients printed below are never updated.
"""
for client in clients:
    
    print(client)
    print(f'\tTotal connected time: {client.total_connected_time:>5}')
    print(f'\tTotal unconnected time: {client.total_unconnected_time:>5}')
    print(f'\tTotal request count: {client.total_request_count:>5}')
    print(f'\tTotal consume time: {client.total_consume_time:>5}')
    print(f'\tTotal usage: {client.total_usage:>5}')
    print()
"""

print("Number or clients:", NUM_CLIENTS)
print('-' * 60)

stats.print_general_stats()
stats.print_detailed_slice_load_stats()

if SETTINGS['plotting_params']['plotting']:
    xlim_left = int(SETTINGS['simulation_time'] * SETTINGS['statistics_params']['warmup_ratio'])
    xlim_right = int(SETTINGS['simulation_time'] * (1 - SETTINGS['statistics_params']['cooldown_ratio'])) + 1

    graph = Graph(base_stations, clients, (xlim_left, xlim_right),
                  ((x_vals['min'], x_vals['max']), (y_vals['min'], y_vals['max'])),
                  output_dpi=SETTINGS['plotting_params']['plot_file_dpi'],
                  scatter_size=SETTINGS['plotting_params']['scatter_size'],
                  output_filename=SETTINGS['plotting_params']['plot_file'])
    graph.draw_all(*stats.get_stats())
    if SETTINGS['plotting_params']['plot_save']:
        graph.save_fig()
    if SETTINGS['plotting_params']['plot_show']:
        graph.show_plot()

sys.stdout = sys.__stdout__
print('Simulation has ran completely and output file created to:', SETTINGS['log_file'])
