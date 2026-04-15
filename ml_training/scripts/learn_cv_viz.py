# STEP 1 — Test your environment
# File: ml_training/scripts/learn_cv_viz.py

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold

print("numpy:", np.__version__)
print("matplotlib:", plt.matplotlib.__version__)
print("All imports OK")