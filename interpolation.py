import torch

# Shift to device if needed
def shift_dev(t, device):
    if t.device != device:
        return t.to(device)
    else:
        return t
    
# Create empty tensors of appropriate size
def alloc(ns, k, y, device):
    lhs = torch.zeros((ns, k+1, k+1), dtype=torch.float32, device=device)
    rhs = torch.zeros((ns, k+1, y.shape[1]), dtype=torch.float32, device=device)
    return lhs, rhs

# Make left hand side of linear system
def make_lhs(lhs, kernel_matrix, batch_indices, k, device):
    lhs[:,:k,:k] = kernel_matrix[batch_indices, :k, :k] + 2*torch.diag(torch.ones(k, device=device))
    lhs[:,k,:] = 1.0
    lhs[:,:,k] = 1.0
    return lhs

# Make right hand side of linear system
def make_rhs(rhs, k, y, neighbors, batch_indices):
    rhs[:,:k] = y[torch.vstack([neighbors[i] for i in batch_indices])]
    rhs[:,k] = 0.0
    return rhs

# Single batched call to torch.linalg.solve!
def solve(lhs, rhs):
    return torch.linalg.solve(lhs, rhs)

# Do efficient batched local interpolation of response variables over the input mesh
def batched_interpolation(pinn_data, y, batch_size=1000, device='cpu'):
    n = len(pinn_data['neighbors'])
    neighbors = pinn_data['neighbors']
    kernel_matrix = shift_dev(pinn_data['kernel_matrix'], device) 
    k = kernel_matrix.shape[1]
    y = shift_dev(y, device)

    coeffs = torch.zeros((n, k, y.shape[1]), dtype=torch.float32, device=device) # Output tensor    

    # Create index lists by neighbor count
    idx_by_count = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[]}

    for i in range(n):
        idx_by_count[len(neighbors[i])].append(i)

    # We can't have more than 10 neighbors
    for k in range(1,10):

        # Chunk up the indices into batches if needed
        batches = len(idx_by_count[k]) // batch_size
        if len(idx_by_count[k]) % batch_size != 0:
            batches += 1

        end = 0
        for _ in range(batches):
            start = end
            end = min(len(idx_by_count[k]), start + batch_size)
            ns = end - start

            batch_indices = idx_by_count[k][start:end]

            # Create LHS tensor and RHS tensor
            lhs, rhs = alloc(ns, k, y, device)          
            lhs = make_lhs(lhs, kernel_matrix, batch_indices, k, device)
            rhs = make_rhs(rhs, k, y, neighbors, batch_indices)

            # Batch solve for the coefficients
            tmp_coeff = solve(lhs, rhs)

            # Store coefficients into output
            idx = 0
            for i in batch_indices:
                coeffs[i, :k, :] = tmp_coeff[idx, :k] # Keep all coefficients
                idx += 1

    return coeffs 


# Unit test

# with open('pinn_data.pkl', 'rb') as f:
#     pinn_data = pickle.load(f)
# with open('y.pkl', 'rb') as f:
#     y = pickle.load(f)    

# y.requires_grad = True
# coeffs = batched_interpolation(pinn_data, y, batch_size=200000, device='cuda')   
# coeffs