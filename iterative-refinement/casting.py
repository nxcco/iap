import numpy as np

# This code helps with simulating different precision on real approximate hardware.
# The itere

# Precision type mapping for efficient lookup
PREC_MAP = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

def to_prec(data, precision_str):
    target_type = PREC_MAP.get(precision_str)

    if isinstance(data, np.ndarray):
        # array case
        return data.astype(target_type)
    else:
        # scalar case
        return target_type(data)

