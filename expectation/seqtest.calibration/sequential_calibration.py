import numpy as np 
import pandas as pd




class e_PIT:
    """
    Sequential e-value test for calibration based on PIT values.
    """

    def __init__(self, alpha: float = 0.05):
        self.z = [] # store PIT values
        self