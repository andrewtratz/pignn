# Script to run inference against test dataset and print out metrics

import torch
from torch import nn
import numpy as np
import os
import datetime

from data import *
from config import *
from loader import *
from interpolation import *
from physics import *
from model import *

# Toggles test dataset versus OOD test dataset
TEST_OOD = True

if TEST_OOD:
    test_loader = MyLoader([i for i in range(496)], shuffle=False, cap_size=CAP_SIZE, path='Datasets/ood/', cache=False)
else:
    test_loader = MyLoader([i for i in range(200)], shuffle=False, cap_size=CAP_SIZE, path='Datasets/test/', cache=False)

device = torch.device('cuda')

model = GCN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
SU = ScaleUp()

def loss_fn(y_pred, y_true):
    losses = []
    overall_loss = torch.tensor(0.0).to(y_pred.device)
    base_fn = torch.nn.MSELoss()
    for i in range(y_pred.shape[1]):
        losses.append(base_fn(y_pred[:,i], y_true[:,i]))
        overall_loss += losses[-1]
    losses.append(overall_loss)
    return losses

predictions = []
observations = []
losses = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        data = batch['instance']
        out = model(data.to(device))
        y_true = SU(data.y)
        y_pred = SU(out)
        predictions.append(y_pred)
        observations.append(y_true)
        loss = loss_fn(out, data.y)
        losses.append(loss[-1])

model = None
test_loader = None

preds = torch.vstack(predictions).to('cpu').numpy()
obs = torch.vstack(observations).to('cpu').numpy()
loss = torch.vstack(losses).to('cpu').numpy()

predictions = None
observations = None
losses = None

print('Best example')
print(np.argmin(loss))

predictions = {'x-velocity': preds[:,0], 'y-velocity': preds[:,1], 'pressure': preds[:,2], 'turbulent_viscosity': preds[:,3]}
observations = {'x-velocity': obs[:,0], 'y-velocity': obs[:,1], 'pressure': obs[:,2], 'turbulent_viscosity': obs[:,3]}


from lips.evaluation.airfrans_evaluation import AirfRANSEvaluation
# Load the required benchmark datasets
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path = LOG_PATH)
benchmark.load(path=DIRECTORY_NAME)


evaluator = AirfRANSEvaluation(config_path = BENCH_CONFIG_PATH,
                               scenario = BENCHMARK_NAME,
                               data_path = DIRECTORY_NAME,
                               log_path = LOG_PATH)

if TEST_OOD:
    observation_metadata = benchmark._test_ood_dataset.extra_data
else:
    observation_metadata = benchmark._test_dataset.extra_data

metrics = evaluator.evaluate(observations=observations,
                             predictions=predictions,
                             observation_metadata=observation_metadata)
print(metrics)