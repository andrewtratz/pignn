# Collection of helper functions used in data preprocessing

import os
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

from config import *

if REFRESH:
    benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                                config_path = BENCH_CONFIG_PATH,
                                benchmark_name = BENCHMARK_NAME,
                                log_path = LOG_PATH)
    benchmark.load(path=DIRECTORY_NAME)


    # Create normalizing constants
    MEANS = {}
    STDS = {}

    for var in ['x-position', 'y-position', 'x-inlet_velocity', 'x-velocity', 'pressure', 'turbulent_viscosity']:
        MEANS[var] = np.mean(benchmark.train_dataset.data[var])
        STDS[var] = np.std(benchmark.train_dataset.data[var])

    MEANS['speed'] = MEANS['x-inlet_velocity']
    STDS['speed'] = STDS['x-inlet_velocity']
    MEANS['position'] = 0.0
    STDS['position'] = 1.0

    for var in ['x-position', 'y-position', 'x-inlet_velocity', 'x-velocity']:
        MEANS.pop(var, None)
        STDS.pop(var, None)

    print(MEANS)
    print(STDS)

    x_means = np.zeros(2, dtype=np.float32)
    x_stds = np.zeros(2, dtype=np.float32)
    y_means = np.zeros(4, dtype=np.float32)
    y_stds = np.zeros(4, dtype=np.float32)

    for i, var in zip(range(2), ['position', 'speed']):        
        x_means[i] = MEANS[var]
        x_stds[i] = STDS[var]

    for i, var in zip(range(4), ['position', 'speed', 'pressure', 'turbulent_viscosity']):        
        y_means[i] = MEANS[var]
        y_stds[i] = STDS[var]

    MEANS['pressure'] = 0.0
    MEANS['turbulent_viscosity'] = 0.0
else:
    MEANS = {'pressure': -395.22959540860137, 'turbulent_viscosity': 0.0008392954292084482, 'speed': 63.15423170302567, 'position': 0.0}
    STDS = {'pressure': 2425.738434726353, 'turbulent_viscosity': 0.0030420989011928183, 'speed': 8.487422521188462, 'position': 1.0}


# Module which converts from raw outputs to scaled outputs
class ScaleUp(nn.Module):
    def forward(self, output):
        y = output.clone()
        speed = y[:,0]*STDS['speed'] + MEANS['speed']
        y[:,0] = (torch.cos(2*torch.pi + output[:,1]))*speed # X velocity
        y[:,1] = (torch.sin(2*torch.pi + output[:,1]))*speed # Y velocity
        y[:, 2] = (y[:,2]*STDS['pressure']) + MEANS['pressure'] # Scaled pressure
        y[:,3] = (y[:,3]*STDS['turbulent_viscosity']) + MEANS['turbulent_viscosity'] # Scaled viscosity
        return y


# Create label and data array for single edge
def make_edge(a, b):
    label = str(a) + '_' + str(b)
    return label, np.array([a,b], dtype=int)


# Line from a to b
def delta_vector(from_v, to_v):
    return to_v-from_v

def angle_off_x_axis(a):
    # Note that both vectors begin at the origin, so we actually want them compared vs. [1,0]
    # if len(a.shape) < 2:
    #     a = np.expand_dims(a, 1)
    if len(a.shape) < 2:
        norm = np.linalg.norm(a)
        out = np.ones_like(norm)
        out = np.arccos(np.divide(a.dot(np.array([1,0])),norm))

    else:
        norm = np.linalg.norm(a, axis=1)
        out = np.ones_like(norm)
        out[np.where(norm > 0)] = np.arccos(np.divide(np.squeeze(a[np.where(norm > 0)]).dot(np.array([1,0])),np.squeeze(norm[np.where(norm > 0)])))
    return out

# Create all edges
def make_edges(sim):
    cells = sim.internal.cells_dict[9]

    edge_dict = {}
    for cell in cells:
        # Create fully connected mesh, including diagonals
        for i in range(0, 4):
            for j in range(0, 4):
                if i == j: 
                    continue
                label, data = make_edge(cell[i], cell[j])
                if label not in edge_dict:
                    edge_dict[label] = data # Push the edge

    # Store de-duplicated bidirectional edges in numpy format
    edge_index = np.zeros((2,len(edge_dict)), dtype=np.intc)
    edge_features = np.zeros((len(edge_dict),2), dtype=np.float32)

    for i, edge in zip(range(0, len(edge_dict)), edge_dict.values()):
        edge_index[:,i] = edge
        edge_features[i,0] = np.sqrt(np.sum((sim.position[edge[0]] - sim.position[edge[1]])**2))
        edge_features[i,1] = angle_off_x_axis(delta_vector(sim.position[edge[0]], sim.position[edge[1]]))
    
    return edge_index, edge_features