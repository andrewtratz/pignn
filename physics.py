import numpy as np
import torch
from torch import nn


# Physical constants - provided by AirfRANS dataset
T = 298.15
MOL = np.array(28.965338e-3) # Air molar weigth in kg/mol
P_ref = np.array(1.01325e5) # Pressure reference in Pa
RHO = P_ref*MOL/(8.3144621*T) # Specific mass of air at temperature T
NU = -3.400747e-6 + 3.452139e-8*T + 1.00881778e-10*T**2 - 1.363528e-14*T**3 # Approximation of the kinematic viscosity of air at temperature T
C = 20.05*np.sqrt(T) # Approximation of the sound velocity of air at temperature T  

# Module to compute physics-informed loss function
# Relies on precomputation of interpolation coefficients per node
# Relies on precomputation of constants used to calculate first and second derivatives based on interpolation results
class PINNLoss(nn.Module):
    def __init__(self, device):
        super(PINNLoss, self).__init__()
        self.device = device

    def forward(self, preds, coeffs, d1terms, d2terms, baseline_err=None, reduce=False):
        nbcoeff = coeffs  #[:, 1:]
        dvx_dx = torch.sum(torch.multiply(nbcoeff[:,:9,0], d1terms[:,:,0]), dim=1)
        dvx_dy = torch.sum(torch.multiply(nbcoeff[:,:9,0], d1terms[:,:,1]), dim=1)
        dvy_dx = torch.sum(torch.multiply(nbcoeff[:,:9,1], d1terms[:,:,0]), dim=1)
        dvy_dy = torch.sum(torch.multiply(nbcoeff[:,:9,1], d1terms[:,:,1]), dim=1)
        dp_dx = torch.sum(torch.multiply(nbcoeff[:,:9,2], d1terms[:,:,0]), dim=1)
        dp_dy = torch.sum(torch.multiply(nbcoeff[:,:9,2], d1terms[:,:,1]), dim=1)
        d2vx_d2x = torch.sum(torch.multiply(nbcoeff[:,:9,0], d2terms[:,:,0]), dim=1)
        d2vx_d2y = torch.sum(torch.multiply(nbcoeff[:,:9,0], d2terms[:,:,1]), dim=1)
        d2vy_d2x = torch.sum(torch.multiply(nbcoeff[:,:9,1], d2terms[:,:,0]), dim=1)
        d2vy_d2y = torch.sum(torch.multiply(nbcoeff[:,:9,0], d2terms[:,:,1]), dim=1)

         # Conservation of mass error term (should be 0)
        mass_err = torch.abs(torch.add(dvx_dx, dvy_dy))

        # Momentum error term (x component)
        lhs = torch.multiply(preds[:, 0], dvx_dx) + torch.multiply(preds[:,1], dvx_dy)
        rhs = -dp_dx + (NU + preds[:,3])*(d2vx_d2x + d2vx_d2y)
        mom_x_err = torch.abs(lhs-rhs)

        # Momentum error term (y component)
        lhs = torch.multiply(preds[:, 0], dvy_dx) + torch.multiply(preds[:,1], dvy_dy)
        rhs = -dp_dy + (NU + preds[:, 3])*(d2vy_d2x + d2vy_d2y)
        mom_y_err = torch.abs(lhs-rhs)

        # Don't penalize if within baseline error level of ground truth data (calibration)
        if baseline_err is not None:
            zero = torch.zeros_like(mass_err, device=mass_err.device)
            base_mass_err = baseline_err['mass_err'].to(self.device)
            base_mom_x_err = baseline_err['mom_x_err'].to(self.device)
            base_mom_y_err = baseline_err['mom_y_err'].to(self.device)
            mass_err = torch.max(zero, mass_err - base_mass_err)
            mom_x_err = torch.max(zero, mom_x_err - base_mom_x_err)
            mom_y_err = torch.max(zero, mom_y_err - base_mom_y_err)

        # Reduce only if requested
        if reduce:
            mass_err = torch.mean(mass_err)
            mom_x_err = torch.mean(mom_x_err)
            mom_y_err = torch.mean(mom_y_err)
        
        return mass_err, mom_x_err, mom_y_err