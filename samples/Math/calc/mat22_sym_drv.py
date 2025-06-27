# -*- coding: utf-8 -*-
# @file mat22_sym_drv.py
# @brief Compute the gradient with respect to a 2x2 symmetric matrix C.
# @author sailing-innocent
# @date 2025-02-23
# @version 1.0
# ---------------------------------

import numpy as np 
import unittest 
from numpy.testing import assert_allclose

def Cinv(C: np.ndarray) -> np.ndarray:
    """
    Compute the inverse of a 2x2 symmetric matrix C.
    
    Args:
        C (np.ndarray): A 2x2 symmetric matrix where C[0,0]=C00, C[0,1]=C[1,0]=C01, C[1,1]=C11.
    
    Returns:
        np.ndarray: A 2x2 symmetric matrix representing the inverse of C.
    """
    C00, C01, C11 = C[0, 0], C[0, 1], C[1, 1]
    Delta = C00 * C11 - C01**2
    return np.array([[C11, -C01], [-C01, C00]]) / Delta

def dLdCinv(C, dCinv):
    """
    Compute the gradient with respect to C given the gradient dCinv
    with respect to the inverse of C, for a 2x2 symmetric matrix C.
    
    Args:
        C (np.ndarray): A 2x2 symmetric matrix where C[0,0]=C00, C[0,1]=C[1,0]=C01, C[1,1]=C11.
        dCinv (np.ndarray): A 2x2 symmetric matrix of gradients with respect to C⁻¹,
                            with elements dCinv[0,0]=dC⁻¹₀₀, dCinv[0,1]=dCinv[1,0]=dC⁻¹₀₁, and dCinv[1,1]=dC⁻¹₁₁.

    Returns:
        np.ndarray: A 2x2 symmetric matrix representing the gradient with respect to C.
    
    The derivatives are computed as:
    
        Δ = C00 * C11 - C01²

        dC00 = (2 * C01 * C11 * dC⁻¹₀₁ - C11² * dC⁻¹₀₀ - C01² * dC⁻¹₁₁) / Δ²
        
        dC01 = (C00 * C01 * dC⁻¹₁₁ + C01 * C11 * dC⁻¹₀₀ -
                (C00 * C11 + C01²) * dC⁻¹₀₁) / Δ²
        
        dC11 = (2 * C00 * C01 * dC⁻¹₀₁ - C00² * dC⁻¹₁₁ - C01² * dC⁻¹₀₀) / Δ²
    """
    # Extract matrix elements.
    C00 = C[0, 0]
    C01 = C[0, 1]
    C11 = C[1, 1]
    
    # Compute the determinant Delta and its square.
    Delta = C00 * C11 - C01**2
    Delta2 = Delta**2

    # Extract gradient elements of dCinv.
    dCinv00 = dCinv[0, 0]
    dCinv01 = dCinv[0, 1]  # symmetry: same as dCinv[1, 0]
    dCinv11 = dCinv[1, 1]
    
    # Compute the derivatives as derived above.
    dC00 = (2 * C01 * C11 * dCinv01 - C11**2 * dCinv00 - C01**2 * dCinv11) / Delta2
    dC01 = (C00 * C01 * dCinv11 + C01 * C11 * dCinv00 - (C00 * C11 + C01**2) * dCinv01) / Delta2
    dC11 = (2 * C00 * C01 * dCinv01 - C00**2 * dCinv11 - C01**2 * dCinv00) / Delta2
    
    # Assemble and return the resulting matrix.
    return np.array([[dC00, dC01],
                     [dC01, dC11]])


def dLdCinv_np(C: np.array, dCinv: np.array) -> np.array:
    """
    Compute dL/dC given dL/dCinv
    for a 2×2 symmetric matrix, with:
    
      dC00 = (2*C01*C11*dCinv01 - C11**2*dCinv00 - C01**2*dCinv11) / Delta^2,
      dC01 = (C01*C11*dCinv00 + C00*C01*dCinv11 - (C00*C11 + C01**2)*dCinv01) / Delta^2,
      dC11 = (2*C00*C01*dCinv01 - C00**2*dCinv11 - C01**2*dCinv00) / Delta^2,
    
    where Delta = C00 * C11 - C01^2.
    """
    assert C.shape == (2, 2)
    assert dCinv.shape == (2, 2)
    C00 = C[0, 0]
    C01 = C[0, 1]
    C11 = C[1, 1]
    Delta = C00 * C11 - C01**2
    Delta2 = Delta**2

    # Extract components of dCinv (symmetric assumed)
    dCinv00 = dCinv[0, 0]
    dCinv01 = dCinv[0, 1]  # same as dCinv[1, 0]
    dCinv11 = dCinv[1, 1]

    dC00 = (2 * C01 * C11 * dCinv01 - C11**2 * dCinv00 - C01**2 * dCinv11) / Delta2
    # Note: the (0,1) derivative must be computed carefully.
    dC01 = (C01 * C11 * dCinv00 + C00 * C01 * dCinv11 - (C00 * C11 + C01**2) * dCinv01) / Delta2
    dC11 = (2 * C00 * C01 * dCinv01 - C00**2 * dCinv11 - C01**2 * dCinv00) / Delta2

    return np.array([[dC00, dC01],
                     [dC01, dC11]])

class TestDLdCinv(unittest.TestCase):

    def check_inv(self, C):
        # Compute the inverse of C using our function.
        invC = Cinv(C)
        invC_target = np.linalg.inv(C)
        # Compare the results.
        assert_allclose(invC, invC_target)

    def check_case(self, C, dCinv, tol=1e-6):
        # Compute expected gradient using the closed-form formula:
        # dLdC = -inv(C) * dCinv * inv(C)
        invC = np.linalg.inv(C)
        expected = - invC @ dCinv @ invC

        # Get our implementation's result.
        result = dLdCinv(C, dCinv)
        # Compare both results.
        assert_allclose(result, expected, atol=tol)

    def test_simple_case(self):
        # Use a simple symmetric positive-definite matrix.
        C = np.array([[4.0, 1.0],
                      [1.0, 3.0]])
        self.check_inv(C)
        dCinv = np.array([[0.1, 0.2],
                          [0.2, 0.3]])
        self.check_case(C, dCinv)

    def test_random_case(self):
        # Generate a random symmetric positive-definite 2x2 matrix.
        np.random.seed(42)
        A = np.random.rand(2, 2) + 1.0
        C = A @ A.T  # symmetric, positive-definite
        # Make dCinv an arbitrary symmetric matrix.
        self.check_inv(C)
        
        B = np.random.rand(2, 2)
        dCinv = 0.5 * (B + B.T)
        self.check_case(C, dCinv)

    def test_another_case(self):
        # Another fixed test case.
        C = np.array([[2.0, 0.5],
                      [0.5, 1.5]])
        self.check_inv(C)    
 
        dCinv = np.array([[0.05, -0.1],
                          [-0.1, 0.2]])
        self.check_case(C, dCinv)

if __name__ == '__main__':
    unittest.main()