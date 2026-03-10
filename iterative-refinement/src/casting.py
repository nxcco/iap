import numpy as np

# Utilities for casting data to a specific floating-point precision. Used throughout
# the project to simulate mixed-precision arithmetic by explicitly downgrading arrays
# or scalars to float16/float32/float64 at each operation.

# Precision type mapping for efficient lookup
PREC_MAP = {
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

# cast data to the given precision
def to_prec(data, precision_str):
    target_type = PREC_MAP.get(precision_str)

    if isinstance(data, np.ndarray):
        # array case
        return data.astype(target_type)
    else:
        # scalar case
        return target_type(data)

