import os
import numpy as np
from config import *
from data import *
from dataset import *

# Load benchmark to debug
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path = LOG_PATH)
benchmark.load(path=DIRECTORY_NAME)

# Make train and CV splits
cv_indices = list(np.array(range(1,int(103/5)),dtype=np.intc)*5)
train_indices = []
for i in range(0,103):
    if i not in cv_indices:
        train_indices.append(i)

# Load datasets

import pickle

TRUNCATED = False
REFRESH = True

if os.path.exists('train.pkl') and not REFRESH:
    file = open('train.pkl', 'rb')
    train = pickle.load(file)
    file.close()
    file = open('cv.pkl', 'rb')
    cv = pickle.load(file)
    file.close()
else:

    if TRUNCATED:
        train = AirFransGeo(benchmark.train_dataset, train_indices[:1])
        # cv = AirFransGeo(benchmark.train_dataset, cv_indices[:4])
    else:
        # train = AirFransGeo(benchmark.train_dataset, train_indices)
        # cv = AirFransGeo(benchmark.train_dataset, cv_indices)
        test = AirFransGeo(benchmark._test_dataset, range(34+142, 200), save_path='Datasets/test')
        test_ood = AirFransGeo(benchmark._test_ood_dataset, range(496), save_path='Datasets/ood')

        # Save files
        # file = open(os.path.join('train.pkl'), 'wb')
        # pickle.dump(train, file)
        # file.close()
        # file = open(os.path.join('cv.pkl'), 'wb')
        # pickle.dump(cv, file)
        # file.close()