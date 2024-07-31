import sys
sys.path.insert(0, "/home/giuseppe/PNI/Projects/nifeatures")

from nifeatures.search import TransformerCV
from nilearn.datasets import load_mni152_brain_mask
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import pandas as pd
import numpy as np

# Generate 100 simulated fMRI activation maps;
MNI = load_mni152_brain_mask(resolution=2)
template_data = MNI.get_fdata()
data = []
target = []

for sample in range(100):
    noisy_data = template_data.copy()
    weight = np.random.normal(0, 1, 1)[0]
    noisy_data[noisy_data == 1] = np.random.normal(0, 1, np.sum(template_data == 1)) * weight
    if sample == 0:
        data = noisy_data.flatten()
    else:
        data = np.vstack((data, noisy_data.flatten()))
    
    target.append(weight)

model = Ridge
mask = load_mni152_brain_mask(resolution=2)

params = {
    "trf__n_peaks": [100, 200],
    "model__alpha": [0.5, 1.5]
}

# Fit the model;
X_train, X_test, y_train, y_test = train_test_split(data, target)
fit = TransformerCV(model, params, transform=True, mask=mask, cv=2, n_jobs=10).fit(X_train, y_train)

print(fit.models)
