import numpy as np


def normalize_V(x, min_value, max_value):
    return (x - min_value) / (max_value - min_value)


def precise_env_name(problem):
    mapping = {
        "pendulum": "Pendulum-v0",
        "halfcheetah": "HalfCheetah-v2",
        "hopper": "Hopper-v2",
        "walker": "Walker2d-v2"}
    return mapping[problem]
