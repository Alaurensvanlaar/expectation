import numpy as np 
import pandas as pd



# Here @Alaurensvanlaar will make the parametric and non-parametric sequential e-value tests for calibration based on PIT values
class e_PIT:
    """
    Sequential e-value test for calibration based on PIT values.
    """

    def __init__(self, alpha: float = 0.05):
        self.z = [] # store PIT values
        self