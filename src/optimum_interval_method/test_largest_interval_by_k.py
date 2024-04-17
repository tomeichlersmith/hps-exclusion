from ._oim import largest_intervals_by_k

import numpy as np

def test_one_d():
    np.testing.assert_array_equal(
        largest_intervals_by_k(np.array([0.1,0.2,0.84])),
        np.array([0.84-0.2, 1.0-0.2, 1.0-0.1, 1.0])
    )
