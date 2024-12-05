
# Dataset preparation 

import torch
import os
import pickle
import pyvista as pv
import numpy as np

from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, Dataset
from scipy.spatial.distance import cdist
from airfrans.simulation import Simulation

from config import *
from data import *
from physics import *
from interpolation import *

# Custom dataset class which performs preprocessing. Pre-computed data is pickled and stored.
class AirFransGeo():
    def __init__(self, dataset, indices, max_neighbors=9, save_path='Datasets/traincv/', device='cuda'):
        self.dataset = dataset
        self.indices = indices
        self.data = []
        self.pinn_data = []

        for i in tqdm(indices):

            sim = Simulation(DIRECTORY_NAME, self.dataset.extra_data['simulation_names'][i,0])

            n = sim.position.shape[0]
            # Additional data used in computation of PINN loss
            neighbor_list = [] # List of adjacent nodes
            kernel_matrix = torch.zeros(n, max_neighbors+1, max_neighbors+1, dtype=torch.float32)
            # shift = torch.zeros((n, 2), dtype=torch.float32) # Shift factors for interpolation
            # scale = torch.zeros((n, 2), dtype=torch.float32)  # Scale factors for interpolation
            d1terms = torch.zeros((n, max_neighbors, 2), dtype=torch.float32)  # First derivative multiplication constants
            d2terms = torch.zeros((n, max_neighbors, 2), dtype=torch.float32) # Second derivative multiplication constants
            true_errors = torch.zeros((n, 3), dtype=torch.float32)

            # sim = extract_dataset_by_simulation('sim', self.dataset, i)
            # global_x = sim.input_velocity # These are global inputs for each node in the mesh
            inlet_speed = np.linalg.norm(sim.input_velocity, axis=1)
            inlet_theta = angle_off_x_axis(sim.input_velocity)

            # Negate the angles if y < 0
            inlet_theta[np.where(sim.input_velocity[:,1] < 0)] *= -1.0

            # X position
            position = (sim.position - MEANS['position']) / STDS['position']

            # global_x = (global_x - MEANS['speed']) / STDS['speed']
            inlet_speed = (inlet_speed - MEANS['speed']) / STDS['speed']

            # Find closest airfoil points
            surface = np.hstack([sim.position[sim.surface], sim.normals[sim.surface]])
            dists = cdist(sim.position, surface[:,:2], metric='euclidean')
            best_idx = np.argmin(dists,axis=1).T.tolist()
            closest_surfaces = np.take(surface, best_idx, axis=0)

            # Vector to closest airfoil point
            vector_to_surface = delta_vector(sim.position, closest_surfaces[:,:2])
            vector_to_surface = (vector_to_surface - MEANS['position']) / STDS['position']

            # Angle (relative to x-axis) to closest airfoil point
            surface_theta = angle_off_x_axis(delta_vector(sim.position, closest_surfaces[:,:2]))

            # Airfoil distance
            surface_distance = (sim.sdf - MEANS['position']) / STDS['position']

            # Rotate normal vector 90 degrees, take the angle in the positive x direction
            # For some reason this feature massively reduced model power - could be finding a local minimum with this

            # rotated_normal = rotate(closest_surfaces[:,2:], -np.pi/2)
            # rotated_normal[np.where(rotated_normal[:,0]<0)] = rotate(rotated_normal[np.where(rotated_normal[:,0]<0)], np.pi)
            # assert(np.min(rotated_normal[:,0]) >= 0.0)
            # flow_theta = angle_off_x_axis(rotated_normal)
            # flow_theta[np.where(rotated_normal[:,1] < 0)] = -1 * np.abs(flow_theta[np.where(rotated_normal[:,1] < 0)])

            # Is_airfoil
            is_airfoil = sim.surface.astype(np.float32)

            # Y Outlet speed
            outlet_speed = np.linalg.norm(sim.velocity, axis=1)
            outlet_speed = (outlet_speed - MEANS['speed']) / STDS['speed']

            # Y Outlet theta
            outlet_theta = angle_off_x_axis(sim.velocity)
            outlet_theta[np.where(sim.velocity[:,1] < 0)] *= -1.0

            # Y Pressure
            outlet_pressure = (sim.pressure - MEANS['pressure']) / STDS['pressure']

            # Y Turb
            outlet_turb = (sim.nu_t - MEANS['turbulent_viscosity']) / STDS['turbulent_viscosity']

            # X and Y coordinates of each point as well as normals (when on airfoil)
            x = np.hstack([position, np.expand_dims(inlet_speed,1), 
                        np.expand_dims(inlet_theta,1), np.expand_dims(is_airfoil,1),
                         vector_to_surface, np.expand_dims(surface_theta,1), surface_distance]) #np.expand_dims(flow_theta, 1)]) 
            y = np.hstack([np.expand_dims(outlet_speed,1), np.expand_dims(outlet_theta,1), outlet_pressure, outlet_turb])
            edge_index, edge_attr = make_edges(sim)            

            instance = Data(x=torch.from_numpy(x.astype(np.float32)), edge_index=torch.from_numpy(edge_index),
                    edge_attr=torch.from_numpy(edge_attr.astype(np.float32)), y=torch.from_numpy(y.astype(np.float32)), 
                    pos=torch.from_numpy(sim.position.astype(np.float32)))
            # self.data.append(instance)


            # Compute data used for PINN loss function
            surface = torch.where(torch.from_numpy(x)[:,4]!=0.0)[0]
            # Remove surface edges first
            edge_mask = ~torch.isin(torch.from_numpy(edge_index)[:,:], surface)
            non_surface = torch.any(edge_mask, dim=0)
            nonsurface_edge_index = torch.from_numpy(edge_index[:,non_surface])

            # Used https://github.com/ArmanMaesumi/torchrbf/ as reference in constructing
            # Localized interpolation functions to compute partial derivatives at each node
            
            # Iterate over every node
            for k in range(x.shape[0]):
                
                if is_airfoil[k] == 1: # We won't compute for nodes on the airfoil surface
                    neighbor_list.append(torch.empty(0))
                    true_errors[k] = 0.0
                else:                   

                    # Get all neighbors
                    neighbor_indices = torch.gather(nonsurface_edge_index, 1, 
                                    torch.unsqueeze(torch.where(nonsurface_edge_index[0,:]==k)[0], 0).repeat(2,1)).numpy()[1]

                    # Calculate distances to all neighbors
                    n_x = torch.unsqueeze(torch.from_numpy(sim.position[k]),0)
                    neighbor_pos = torch.from_numpy(sim.position[neighbor_indices])
                    # r = torch.squeeze(torch.cdist(n_x, neighbor_pos))
                    
                    neighbors = neighbor_pos.shape[0]+1
                    aug_neighbor_indices = torch.cat((torch.tensor([i]), torch.tensor(neighbor_indices))).type(torch.IntTensor)
                    aug_neighbors = torch.cat((n_x, neighbor_pos))
                    neighbor_list.append(aug_neighbor_indices)
                    dist_matrix = torch.cdist(aug_neighbors, aug_neighbors, compute_mode="use_mm_for_euclid_dist")
                    r = dist_matrix[0,:neighbors]

                    # Multiquadratic kernel matrix
                    kernel_matrix[k][:neighbors,:neighbors] = -torch.sqrt(dist_matrix**2 + 1)[:neighbors,:neighbors] 
                    # kernel_matrix[k][:neighbors,:neighbors] += torch.diag(torch.tensor([1.0])) # Smoothing - weird since it just eliminates self terms
                    
                    kernel_matrix[k][neighbors,:] = 1.0 # Intercept terms - now in 0 row and column
                    kernel_matrix[k][:,neighbors] = 1.0
                    kernel_matrix[k][neighbors:, neighbors:] = 0.0

                    # Calc first derivative multipliers
                    delts = torch.cat((torch.zeros(1, 2), n_x - neighbor_pos))
                    d1terms[k, :neighbors, :] = -delts / (torch.unsqueeze((torch.sqrt(r**2 + 1)), 1))

                    # Calc second derivative multipliers
                    d2terms[k, :neighbors, :] = delts**2 / torch.unsqueeze((r**2+1)**(1.5), 1) - \
                                                            torch.unsqueeze(1/(r**2+1)**(0.5), 1)

            # Cache the data for future use
            self.pinn_data.append({})      
            self.pinn_data[-1]['neighbors'] = neighbor_list
            self.pinn_data[-1]['kernel_matrix'] = kernel_matrix
            self.pinn_data[-1]['d1terms'] = d1terms
            self.pinn_data[-1]['d2terms'] = d2terms  

            y = torch.hstack([torch.from_numpy(sim.velocity), torch.from_numpy(sim.pressure)])

            # Keep track of the "true" coefficients and baseline errors based on the target outputs, for calibration
            coeffs = batched_interpolation(self.pinn_data[-1], y, batch_size=200000, device='cuda')
            self.pinn_data[-1]['true_coeffs'] = coeffs.detach().cpu() 

            PL = PINNLoss(device='cpu')
            preds = torch.hstack([torch.from_numpy(sim.velocity), torch.tensor(torch.from_numpy(sim.pressure)), torch.from_numpy(sim.nu_t)])
            mass_err, mom_x_err, mom_y_err = PL.forward(preds, self.pinn_data[-1]['true_coeffs'], self.pinn_data[-1]['d1terms'], self.pinn_data[-1]['d2terms'])   #preds, coeffs, d1terms, d2terms
            self.pinn_data[-1]['mass_err'] = mass_err
            self.pinn_data[-1]['mom_x_err'] = mom_x_err
            self.pinn_data[-1]['mom_y_err'] = mom_y_err

            with open(save_path + str(i) + '.pkl', 'wb') as handle:
                pickle.dump({'instance': instance, 'pinn_data': self.pinn_data[-1]}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def len(self):
        return len(self.indices)

    def get(self,index):
        return self.data[index]