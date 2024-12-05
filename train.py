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

if not os.path.exists('models'):
    os.mkdir('models')

# Load benchmark to debug
# from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

# benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
#                             config_path = BENCH_CONFIG_PATH,
#                             benchmark_name = BENCHMARK_NAME,
#                             log_path = LOG_PATH)
# benchmark.load(path=DIRECTORY_NAME)

current_time = datetime.datetime.now()
time_path = str(current_time.month) + '_' + str(current_time.day) + '_' + str(current_time.hour) + '_' + str(current_time.minute) 
save_path = os.path.join('models', time_path)
os.mkdir(save_path)

# Make train and CV splits
cv_indices = list(np.array(range(1,int(103/5)),dtype=np.intc)*5)
train_indices = []
for i in range(0,103):
    if i not in cv_indices:
        train_indices.append(i)

train_loader = MyLoader(train_indices, shuffle=SHUFFLE, cap_size=CAP_SIZE)
cv_loader = MyLoader(cv_indices, shuffle=False, cap_size=CAP_SIZE)

device = torch.device('cuda')

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

model = GCN()
if LOAD_MODEL:
    print("Loading existing weights!")
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

model.train()
# scaler = torch.tensor([SPEED_SCALE, SPEED_SCALE, PRESS_SCALE, TURB_SCALE]).to(device)
# x_scaler.to(device)
# loss_fn = torch.nn.MSELoss()

def loss_fn(y_pred, y_true):
    losses = []
    overall_loss = torch.tensor(0.0).to(y_pred.device)
    base_fn = torch.nn.MSELoss()
    for i in range(y_pred.shape[1]):
        losses.append(base_fn(y_pred[:,i], y_true[:,i]))
        overall_loss += losses[-1]
    losses.append(overall_loss)
    return losses

if PINN_LOSS_ON:
    losstypes = ['loss_speed', 'loss_theta', 'loss_press', 'loss_turb', 'mass_err', 'mom_x_err', 'mom_y_err', 'loss_train', 'combo_train']
else:
    losstypes = ['loss_speed', 'loss_theta', 'loss_press', 'loss_turb', 'loss_train']

def train_one_epoch(model, optimizer, train_loader, device, scaler, losstypes, loss_fn, epoch):

    # Initiatlize losses to empty
    PL = PINNLoss(device)
    SU = ScaleUp()
    losses={}
    

    for loss in losstypes:
        losses[loss] = []

    print ("Training")
    for batch in tqdm(train_loader):
        optimizer.zero_grad()

        data = batch['instance']
        if PINN_LOSS_ON:
            pinn_data = batch['pinn_data']

        out = model(data.to(device))
        # y_true = torch.divide(batch.y, scaler)
        y_true = data.y
        loss_speed, loss_theta, loss_press, loss_turb, loss_train = loss_fn(out, y_true)

        if PINN_LOSS_ON:
            # Combine loss with physics-informed loss term
            scaled_output = SU(out)
            pred_coeffs = batched_interpolation(pinn_data, scaled_output, batch_size=200000, device=device)
            mass_err, mom_x_err, mom_y_err = PL.forward(scaled_output, pred_coeffs, pinn_data['d1terms'].to(device), pinn_data['d2terms'].to(device),
                                                        pinn_data, reduce=True)        
            avg_pinn_err = mass_err + mom_x_err + mom_y_err
            combo_train = loss_train + avg_pinn_err*LAMBDA

        for loss in losstypes:
            losses[loss].append(eval(loss))

        if PINN_LOSS_ON:
            combo_train.backward()
        else:
            loss_train.backward()
        optimizer.step()

    # Wrap up epoch
    avg_losses = {}
    for loss in losstypes:
        avg_losses[loss] = torch.mean(torch.tensor(losses[loss])).detach().cpu().item()
    print("Epoch " + str(epoch))
    loss_str = "T Losses "
    for loss in losstypes:
        loss_str += ' ' + loss + ':' + "{:10.4f}".format(avg_losses[loss])
    # print("Losses " + [[loss, avg_losses[loss]].join(': ') for loss in losstypes].join(' '))
    print(loss_str)
    for loss in losstypes:
        writer.add_scalar("train/" + loss, avg_losses[loss], epoch)

def validation_loop(model, cv_loader, device, scaler, losstypes, loss_fn, best_loss):

    PL = PINNLoss(device)
    SU = ScaleUp()

    # Initiatlize losses to empty
    losses = {}
    for loss in losstypes:
        losses[loss] = []

    print("CV")
    with torch.no_grad():
        for batch in tqdm(cv_loader):
            data = batch['instance']
            if PINN_LOSS_ON:
                pinn_data = batch['pinn_data']
            out = model(data.to(device))
            # y_true = torch.divide(batch.y, scaler)
            y_true = data.y
            loss_speed, loss_theta, loss_press, loss_turb, loss_cv = loss_fn(out, y_true)

            
            if PINN_LOSS_ON:
                # Combine loss with physics-informed loss term
                scaled_output = SU(out)
                pred_coeffs = batched_interpolation(pinn_data, scaled_output, batch_size=200000, device=device)
                mass_err, mom_x_err, mom_y_err = PL.forward(scaled_output, pred_coeffs, pinn_data['d1terms'].to(device), pinn_data['d2terms'].to(device),
                                                            pinn_data, reduce=True)        
                avg_pinn_err = mass_err + mom_x_err + mom_y_err
                combo_cv = loss_cv + avg_pinn_err*LAMBDA

            for loss in losstypes:
                losses[loss].append(eval(loss).item())

            if PINN_LOSS_ON:
                if combo_cv.cpu().item() < best_loss:
                    best_loss = combo_cv.cpu().item()
                    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))
            else:
                if loss_cv.cpu().item() < best_loss:
                    best_loss = loss_cv.cpu().item()
                    torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

    # Wrap up CV
    avg_losses = {}
    for loss in losstypes:
        avg_losses[loss] = torch.mean(torch.tensor(losses[loss])).detach().cpu().item()
    loss_str = " C Losses "
    for loss in losstypes:
        loss_str += ' ' + loss + ':' + "{:10.4f}".format(avg_losses[loss])
    print(loss_str)
    for loss in losstypes:
        writer.add_scalar("cv/" + loss, avg_losses[loss], epoch)
    return best_loss

best_loss = 10000.0
for epoch in range(EPOCHS):
    train_one_epoch(model, optimizer, train_loader, device, None, losstypes, loss_fn, epoch)
    cv_losses = losstypes[:-2]
    cv_losses.append('loss_cv')
    if PINN_LOSS_ON:
        cv_losses.append('combo_cv')
    best_loss = validation_loop(model, cv_loader, device, None, cv_losses, loss_fn, best_loss)