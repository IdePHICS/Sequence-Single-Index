
from typing import Union

from itertools import product

def generate_combinations(input_dict: dict) -> list[dict]:
    # Get the lists for each key
    keys = list(input_dict.keys())
    value_lists = [input_dict[key] for key in keys]
    
    # Use product to generate all combinations
    combinations = [dict(zip(keys, combo)) for combo in product(*value_lists)]
    
    return combinations

import torch

def crossing_threshold(data: torch.tensor, threshold: float, time: Union[None, torch.tensor] = None) -> Union[float, int]:
    """
    Find the first time the data is greater(smaller) than the threshold.
    
    data: torch.tensor, shape (..., T) 
    threshold: float"
    time: torch.tensor, shape (T,) or None"
    """
    with torch.no_grad():
        if time is None:
            time = torch.arange(data.shape[-1], requires_grad=False)
        
        index = torch.argmax(
            torch.tensor(data > threshold, dtype=torch.float),
            dim=-1
        )
        # check if index == tensor(0)
        if index == torch.tensor(0):
            return float('+inf')
        crossing_time = time[index]
        return crossing_time

