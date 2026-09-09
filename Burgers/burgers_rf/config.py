import numpy as np
import torch


DTYPE = torch.float64

# PDE parameters
NV = 0.01 / np.pi

# Sampling parameters
M_TRAIN = 3600
M_BD = 500
M_INT = 500
M_TEST = 100
N_TEST_TIMES = 10
HERMGAUSS_ORDER = 80

# Random-feature parameters for Model 1: Gaussian x and Gaussian t
NX = 500
NT = 10
SIGMA_X = 1 / 25
SIGMA_T = 1 / 3
DF_X = 1
DF_T = 1
X_DIST = "Gaussian"
T_DIST = "Gaussian"

# Local enrichment parameters
USE_LOCAL = True
LOCAL_TYPE = "gaussian"
LOCAL_CENTER = 0.0
LOCAL_WIDTH = 0.05

# Training parameters
N_TRIALS = 1
EPOCHS = 1000
LR = 5e-4
WEIGHT_DECAY = 1
IC_BC_WEIGHT = 1e3


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
