import numpy as np 
import pandas as pd
#from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, List
from scipy.stats import beta
import statsmodels.api as sm

# Here @Alaurensvanlaar will make the parametric and non-parametric sequential e-value tests for calibration based on PIT values
class e_PIT:
    """
    Sequential e-value test for calibration based on PIT values.
    """
    def __init__(self, z: np.ndarray):
        self.z: np.ndarray = np.asarray(z, dtype=float)  # ensure NumPy array and store as NumPy array
        self._check(self.z)                   # validate PIT values       
        self.n: int = len(self.z)        # number of PIT values
        self.strategy: str = "None"
        self.e_values: np.ndarray = np.ones(self.n, dtype=float)  # placeholder for e-values
        self.e_process: np.ndarray = np.ones(self.n, dtype=float)  # placeholder for e-process
        #self.strategy.fit(self.z, self.n0)
        #self.e_values = self.strategy.compute_e(self.z)
    
    def fit(self, strategy: str = "beta", n0: int = 10): 

        if strategy not in ["beta", "kernel"]: # Validates strategy parameter       
            raise ValueError(f"strategy must be either 'beta' or 'kernel', got '{strategy}'")
        elif n0 > self.n:
            raise ValueError(f"Warning: Not enough observations to compute initial fit. Need at least n0={self.n0}, but got n={self.n}.")
        elif strategy == "beta":
            self.strategy = strategy
            self.epit = beta_e(z=self.z,n0=n0)
        elif strategy == "kernel":
            self.strategy = strategy
            self.epit = kernel_e(z=self.z,n0=n0)
        self.e_values = self.epit.compute_e(self.z)
        
        return self
        
    def update(self, z_new: np.ndarray) -> None:
        """
        Add new PIT values to the class.

        Parameters:
            z_new (np.ndarray): New PIT values to add.
        """
        if not hasattr(self, "epit"):
            # can only update if self.epit is either an instance of beta_e or kernel_e
            raise ValueError(f"User has to use the fit() before being able to update")

        # update the PIT values and number of observations
        z_new = np.asarray(z_new, dtype=float)
        self._check(z_new)
        self.z = np.concatenate([self.z, z_new])
        self.n = len(self.z)
        # Update the beta_e or kernel_e instance
        self.epit.update(z=self.z)
        # update the e-values
        self.e_values = self.epit.compute_e(self.z)


    def summary(self) -> str:
        """Return a summary of the PIT values."""
        return (
            f"e_PIT Summary:\n"
            f"-------------------------\n"
            f"Number of PIT values: {self.n}\n"
            f"fit : {self.strategy}\n"
            f"Minimum value: {self.z.min():.4f}\n"
            f"Maximum value: {self.z.max():.4f}\n"
            f"Mean value: {self.z.mean():.4f}\n"
            f"-------------------------\n"
        )
    
    def __str__(self) -> str: # defines the string representation which allows to use print function on the class e_PIT 
        return self.summary()

    @staticmethod # static method doesn't receive self
    def _check(z: np.ndarray) -> None:
        """ Check if PIT values are in [0, 1] """
        if np.any((z < 0) | (z > 1)):
            raise ValueError("PIT values must be in the interval [0, 1].")
        
    # EXAMPLE OF PROPERTY
    @property
    def sum_n(self):
        return self.n


class beta_e:
    """
    Sequential e-value test for calibration based on PIT values using a Beta alternative.
    """
    def __init__(self,z:np.ndarray, n0:int):
        self.n0 = n0
        self.n = len(z)
        self.sequential_params: np.ndarray = np.full((self.n, 2), 1,dtype=float)
        """Compute sequential MLE for initial PIT values > n0."""
        for t in range(self.n0, self.n):
            pit_subset = z[:t]
            a_hat, b_hat, _, _ = beta.fit(pit_subset, floc=0, fscale=1) # beta from scipy.stats
            self.sequential_params[t] = [a_hat, b_hat]
        # For stability, the estimates of (a_hat,b_hat) are truncated to lie in [0.001, 100] see technical notes in original paper
        self.sequential_params = np.clip(self.sequential_params, 0.001, 100)

    def compute_e(self,z: np.ndarray) -> np.ndarray:
        """Return the beta e-values."""
        e_values = beta.pdf(z, self.sequential_params[:,0], self.sequential_params[:,1])
        return e_values
    
    def update(self, z: np.ndarray) -> None:
        """update sequential estimates for new indices only."""       
        self.n = len(z)
        old_len = self.sequential_params.shape[0]
        
        # Extend sequential_params array
        new_array = np.full((self.n, 2), np.nan)
        new_array[:old_len, :] = self.sequential_params
        self.sequential_params = new_array
        # Compute sequentially the alpha and beta MLE only for new indices
        for t in range(max(self.n0, old_len), self.n):
            pit_subset = z[:t]
            a_hat, b_hat, _, _ = beta.fit(pit_subset, floc=0, fscale=1)
            self.sequential_params[t] = [a_hat, b_hat]
        #  For stability, the estimates of (a_hat,b_hat) are truncated to lie in [0.001, 100]
        self.sequential_params = np.clip(self.sequential_params, 0.001, 100)


class kernel_e:
    def __init__(self,z:np.ndarray, n0:int):
        self.n0 = n0
        self.n = len(z)
        self.list_denst = [] # list containing all sequential kde estimates
        
        for t in range(self.n0,self.n):
            pit_subset = z[:t] 
            # This currently doesnt implement the same Kde process as the original paper
            # See technical note B.2
            dens = sm.nonparametric.KDEUnivariate(endog=pit_subset).fit(kernel = 'gau',fft=True)
            self.list_denst.append(dens)

    def compute_e(self,z:np.ndarray) -> np.ndarray:
        """Return the kernel e-values"""
        e_values = np.ones(len(z))
        for i, dens in enumerate(self.list_denst):
            e_values[self.n0 + i] = dens.evaluate(z[self.n0 + i])

        return e_values
    
    def update():
        pass


#%%
""""
 sm.nonparametric.KDEUnivariate.fit(kernel = epa) # Epanechnikov kernel simular to that of the paper
 https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kde.KDEUnivariate.fit.html#statsmodels.nonparametric.kde.KDEUnivariate.fit

"""
