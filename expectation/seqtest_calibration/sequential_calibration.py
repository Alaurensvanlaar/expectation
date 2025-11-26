import numpy as np 
import pandas as pd
from pydantic import BaseModel 
from typing import Optional, List
from scipy.stats import beta



# Here @Alaurensvanlaar will make the parametric and non-parametric sequential e-value tests for calibration based on PIT values
class e_PIT:
    """
    Sequential e-value test for calibration based on PIT values.
    """

    def __init__(self, z: np.ndarray,n0: int =10):
        z = np.asarray(z, dtype=float)  # ensure NumPy array
        self._check(z)                   # validate PIT values
        self.z: np.ndarray = z           # store as NumPy array
        self.n: int = len(self.z)        # number of PIT values
        self.n0: int = 10                # minimum number of observations
        self.E_values: Optional[np.ndarray] = None  # placeholder for e-values
        self.E_process: Optional[np.ndarray] = None  # placeholder for e-process
    

    def _check(self, z: np.ndarray) -> None:
        """ Check if PIT values are in [0, 1] """
        if np.any((z < 0) | (z > 1)):
            raise ValueError("PIT values must be in the interval [0, 1].")
    
    
    def add_value(self, z_new: np.ndarray) -> None:
        """
        Add new PIT values to the class.

        Parameters:
            z_new (np.ndarray): New PIT values to add.
        """
        z_new = np.asarray(z_new, dtype=float)
        self._check(z_new)
        self.z = np.concatenate([self.z, z_new])
        self.n = len(self.z)
    
    def summary(self) -> str:
        """Return a summary of the PIT values."""
        return (
            f"e_PIT Summary:\n"
            f"Number of PIT values: {self.n}\n"
            f"Minimum value: {self.z.min():.4f}\n"
            f"Maximum value: {self.z.max():.4f}\n"
            f"Mean value: {self.z.mean():.4f}"
        )
    
    def __str__(self) -> str: # defines the string representation which allows to use print function on the class e_PIT 
        return self.summary()
    

class beta_e(e_PIT):
    """
    Sequential e-value test for calibration based on PIT values using a Beta alternative.
    """

    def __init__(self, z: np.ndarray, n0: int = 10):
        super().__init__(z, n0)
        self.sequential_params: np.ndarray = np.full((self.n, 2), np.nan)  # store sequential (a,b)
        
        if self.n >= self.n0: # compute only once for initial 
            self._compute_initial_fit()  z

    def _compute_initial_fit(self) -> None:
        """Compute sequential MLE for initial PIT values ≥ n0."""
        for t in range(self.n0, self.n):
            pit_subset = self.z[:t+1]
            a_hat, b_hat, _, _ = beta.fit(pit_subset, floc=0, fscale=1)
            self.sequential_params[t] = [a_hat, b_hat]

    def add_value(self, z_new: np.ndarray) -> None:
        """Add new PIT values and update sequential estimates for new indices only."""
        super().add_value(z_new)  # updates self.z and self.n
        old_len = self.sequential_params.shape[0]
        # Extend sequential_params array
        new_array = np.full((self.n, 2), np.nan)
        new_array[:old_len, :] = self.sequential_params
        self.sequential_params = new_array

        # Compute sequential MLE only for new indices
        for t in range(max(self.n0, old_len), self.n):
            pit_subset = self.z[:t+1]
            a_hat, b_hat, _, _ = beta.fit(pit_subset, floc=0, fscale=1)
            self.sequential_params[t] = [a_hat, b_hat]

    def e_values(self) -> np.ndarray:
        """Compute sequential e-values (vectorized)."""
        e_vals = np.ones(self.n)
        for t in range(self.n0, self.n):
            pit_subset = self.z[:t+1]
            a_hat, b_hat = self.sequential_params[t]
            # vectorized likelihood ratio
            e_vals[t] = np.prod(beta.pdf(pit_subset, a_hat, b_hat))
        return e_vals

    def summary(self) -> str:
        return (
            f"beta_e summary:\n"
            f"Number of PIT values: {self.n}\n"
            f"n0 (min obs for estimation): {self.n0}\n"
            f"Latest estimated a, b: {self.sequential_params[self.n - 1]}\n"
            f"PIT min: {self.z.min():.4f}, PIT max: {self.z.max():.4f}, PIT mean: {self.z.mean():.4f}"
        )
