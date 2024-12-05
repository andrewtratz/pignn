# Script to preprocess test and OOD-test datasets

from lips import get_root_path
from lips.dataset import airfransDataSet
from lips.dataset.airfransDataSet import AirfRANSDataSet
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from torch import nn

from airfrans.simulation import Simulation
import pyvista as pv

import xgboost as xgb
import pickle

import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

from config import *
from data import *
from dataset import *

LIPS_PATH = get_root_path()
DIRECTORY_NAME = '../Airfrans/Dataset'
BENCHMARK_NAME = "Case1"
LOG_PATH = LIPS_PATH + "lips_logs.log"
BENCH_CONFIG_PATH = os.path.join("../Kit", "airfoilConfigurations","benchmarks","confAirfoil.ini") #Configuration file related to the benchmark
SIM_CONFIG_PATH = os.path.join("../Kit", "airfoilConfigurations","simulators","torch_fc.ini") #Configuration file re

benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path = LOG_PATH)
benchmark.load(path=DIRECTORY_NAME)

AirFransGeo(benchmark._test_dataset, [i for i in range(benchmark._test_dataset.extra_data['simulation_names'].shape[0])][:], save_path='Datasets/test/')
AirFransGeo(benchmark._test_ood_dataset, [i for i in range(benchmark._test_ood_dataset.extra_data['simulation_names'].shape[0])][:], save_path='Datasets/ood/')