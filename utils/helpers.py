import numpy as np

def worker_init_fn(worker_id):
    """Initialize DataLoader worker seed"""
    np.random.seed()