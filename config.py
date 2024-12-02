import os
from lips import get_root_path

# Model load
LOAD_MODEL = True
MODEL_PATH = 'models/11_25_0_10/best_model.pth'

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
CAP_SIZE = 2000
BATCH_SIZE = 1
EPOCHS = 500

# Hyperparams
PINN_LOSS_ON = False
LAMBDA = 5.0   




