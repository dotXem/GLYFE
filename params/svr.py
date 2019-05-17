parameters = {
    "hist": 60,
    "kernel": "rbf",
    "C": [1e-2, 1e1],
    "epsilon": 1e-3,
    "gamma": [1e-5, 1e-2],
    "shrinking": False,
}

search = {
    "C": ["logarithmic", 4, 4],
    "gamma": ["logarithmic", 4, 4],
}
