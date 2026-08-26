"""Verify that core ML and backend packages import and C-extensions work."""

import numpy as np
import pandas as pd  # noqa: F401
import scipy.sparse
import fastapi  # noqa: F401
import redis  # noqa: F401

_ = np.array([[1.0, 2.0], [3.0, 4.0]])
_ = scipy.sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])

print("Environment Verification Passed!")
