import numpy as np
from scipy.stats import norm

def expected_improvement(mu, sigma, f_max, alpha=0.5):

    # Return 0 where sigma is zero
    if np.all(sigma == 0):
        return np.zeros_like(mu)

    # Adjust exploration-exploitation trade-off with alpha
    xi = alpha * sigma.max()  # Adjust xi based on the range of sigma for better scaling

    # Calculate Z and EI only for non-zero sigma values
    Z = (mu - f_max - xi) / sigma
    ei = (mu - f_max - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)

    # Set EI to 0 where sigma is zero
    ei[sigma == 0] = 0.0

    return ei


def random_selection_from_top_EIvalues(arr, n=30, seed=None):

    if seed is not None:
        np.random.seed(seed)
    
    # Ascending order
    unique_values = np.unique(arr)[::-1]
    
    selected_indices = []
    
    for val in unique_values:
        indices = np.where(arr == val)[0]
        needed = n - len(selected_indices)
        
        if len(indices) >= needed:

            chosen = np.random.choice(indices, size=needed, replace=False)
            selected_indices.extend(chosen)
            break 
        else:
            selected_indices.extend(indices)
    
    return np.array(selected_indices)
