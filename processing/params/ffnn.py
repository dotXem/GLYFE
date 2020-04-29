import torch

parameters = {
    "hist": 180,
    "n_out": 1,
    "hidden": [128, 64, 32, 16],
    "cell_type": torch.nn.SELU,

    # training
    "dropout": 0.0,
    "l2": 0.0,
    "epochs": 1000,
    "batch_size": 1500,
    "lr": [1e-4, 1e-2],  # learning rate
    "patience": 100,
}

search = {
    "lr": ["logarithmic", 3, 3],
}
