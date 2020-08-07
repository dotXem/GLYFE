parameters = {
    "hist": 180,
    "kernel": "rbf",
    "C": [1e-2, 1e3],
    "epsilon": [1e-3, 1e-1],
    "gamma": [1e-5, 1e-3],
}

search = {
    "C": ["logarithmic", 3, 3],
    "epsilon": ["logarithmic", 3, 3],
    "gamma": ["logarithmic", 3, 3],
}
