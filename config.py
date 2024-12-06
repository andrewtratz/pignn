# Configuration parameters

import os
from lips import get_root_path

# Model load
LOAD_MODEL = False
MODEL_PATH = 'redux_model.pth'

# Simulation paths
LIPS_PATH = get_root_path()
DIRECTORY_NAME = '../Airfrans/Dataset'
BENCHMARK_NAME = "Case1"
LOG_PATH = LIPS_PATH + "lips_logs.log"
BENCH_CONFIG_PATH = os.path.join("../Kit", "airfoilConfigurations","benchmarks","confAirfoil.ini") #Configuration file related to the benchmark
SIM_CONFIG_PATH = os.path.join("../Kit", "airfoilConfigurations","simulators","torch_fc.ini") #Configuration file re

# Data params
REFRESH = False

# Model params
FEATS = 9
NODES = 16
OUTPUTS = 4
CONV_LAYERS = 8
LAYER_REPEATS = 4
activation = 'GELU'

# Training params
CAP_SIZE = 1000
SHUFFLE = True
BATCH_SIZE = 1
EPOCHS = 500

# Hyperparams
PINN_LOSS_ON = False
LAMBDA = 5.0
LR = 0.0005   




