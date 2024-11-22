import torch

# Shift to device if needed
def shift_dev(t, device):
    if t.device != device:
        return t.to(device)
    else:
        return t

# Do efficient batched local interpolation of response variables over the input mesh

def batched_interpolation(pinn_data, y, batch_size=1000, device='cpu'):
    n = len(pinn_data['neighbors'])
    neighbors = pinn_data['neighbors']
    kernel_matrix = shift_dev(pinn_data['kernel_matrix'], device) 
    k = kernel_matrix.shape[1]
    y = shift_dev(y, device)

    coeffs = torch.zeros((n, k, y.shape[1]), dtype=torch.float32, device=device) # Output tensor    

    # Batch everything up
    # batches = n // batch_size
    # if n % batch_size != 0:
    #     batches += 1

    end = 0
    while(True):
        # start = batch_num*batch_size
        # end = min(n, start+batch_size)
        start = end
        end = min(n, start+batch_size)

        # Each batch must match on # of neighbors dimension
        k = len(neighbors[start])

        # Avoid situations where start has zero dimension k
        if k == 0:
            if start == n-1:
                break
            else:
                end = start + 1
                continue

        # Set up LHS tensor
        # Count non-surface nodes in batch
        ns = 0
        for i in range(start, end):
            if len(neighbors[i]) == k:
                ns += 1
            if len(neighbors[i]) > 0 and len(neighbors[i]) != k:
                end = i # Set end to where our k neighbors change quantity
                break

        # Create LHS tensor and RHS tensor
        lhs = torch.zeros((ns, k+1, k+1), dtype=torch.float32, device=device)
        rhs = torch.zeros((ns, k+1, y.shape[1]), dtype=torch.float32, device=device)
        idx = 0
        for i in range(start, end):
            if len(neighbors[i]) == 0:
                continue
            lhs[idx,:k,:k] = kernel_matrix[i, :k, :k] + 2*torch.diag(torch.ones(k, device=device)) # Smoothing eliminates singularity
            lhs[idx,k,:] = 1.0
            lhs[idx,:,k] = 1.0 
            # lhs[idx] = kernel_matrix[i, :k, :k] + torch.diag(torch.ones(k, device=device)*-1.0) # Smoothing eliminates singularity

            # rank = torch.linalg.matrix_rank(lhs[idx])
            # if rank != k:
            #     print("ruh roh!")

            y_neighbor = y[neighbors[i].type(torch.IntTensor)]
            rhs[idx, :k] = y_neighbor[:k]
            rhs[idx, k] = 0.0
            idx += 1

        # Batch solve for the coefficients
        tmp_coeff = torch.linalg.solve(lhs, rhs)

        # Store coefficients into output
        idx = 0
        for i in range(start, end):
            if len(neighbors[i]) == 0:
                continue
            coeffs[i, :k, :] = tmp_coeff[idx, :k] # Keep all coefficients
            idx += 1

        # Break if finished
        if end == n-1:
            break

    return coeffs 


# Unit test

# with open('pinn_data.pkl', 'rb') as f:
#     pinn_data = pickle.load(f)
# with open('y.pkl', 'rb') as f:
#     y = pickle.load(f)    

# y.requires_grad = True
# coeffs = batched_interpolation(pinn_data, y, batch_size=200000, device='cuda')   
# coeffs