import torch
import numpy as np
import os
import datetime

from data import *
from config import *
from loader import *
from interpolation import *
from physics import *
from model import *
# from train import ScaleUp


test_loader = MyLoader([i for i in range(200)], shuffle=False, cap_size=CAP_SIZE, path='Datasets/test/', cache=False)

device = torch.device('cuda')

model = GCN()
model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
model.to(device)
SU = ScaleUp()

predictions = []
observations = []
with torch.no_grad():
    for batch in tqdm(test_loader):
        data = batch['instance']
        out = model(data.to(device))
        y_true = SU(data.y)
        y_pred = SU(out)
        predictions.append(y_pred)
        observations.append(y_true)

preds = torch.vstack(predictions).to('cpu').numpy()
obs = torch.vstack(observations).to('cpu').numpy()

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

observation_metadata = benchmark._test_dataset.extra_data
metrics = evaluator.evaluate(observations=observations,
                             predictions=predictions,
                             observation_metadata=observation_metadata)
print(metrics)