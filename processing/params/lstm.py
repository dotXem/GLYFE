parameters = {
    "hist": 180,

    # model hyperparameters
    "hidden": [256, 256],

    # training hyperparameters
    # "dropout": 0.0,
    # "recurrent_dropout": 0.3, #0.9, #0.5,
    "dropi": 0.0,
    "dropw": 0.0,
    "dropo": 0.0,
    "epochs": 5000,
    "batch_size": 50,
    "lr": [1e-4, 1e-3],
    "l2": 1e-4,
    "patience": 50,
}

search = {
    "lr": ["logarithmic", 3, 3],
}
