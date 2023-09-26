NiFeatures
==========================

## Feature transformation for Machine Learning in Python and Scikit-Learn with neuroimaging data

### What is the purpose of this package?
NiFeatures is a collection of Scikit-Learn compatible transformers, tools that clean, reduce, expand or generate
feature representations for machine learning models (Click [here](https://scikit-learn.org/stable/data_transforms.html) 
for a more detailed description of scikit-learn transformers).

### What is currently implemented in the package?
#### 1. Displacement Invariant transformer
Spatial patterns of neural activity can vary widely between individuals due to genetic and experiential factors,
and, in neuroimaging analyses, comparability across individuals is further compromised by various technical factors
(e.g. inaccurate spatial standardization). Yet, interindividual voxel-by-voxel correspondence is an implicit
assumption in most type of analyses, including brain-based predictive modelling. We address this issue with our 
Displacement Invariant Transformer (DIT), which generates new meaningful features by relaxing this assumption.

Check out the concept behind the DIT with our [IBRO2023 poster](https://twitter.com/g_gallitto/status/1700817459681361984).

#### 2. TransformerCV
An hyperparameter search class tailored for our Displacement Invariant Transformer. Give a look at how it works in
our example notebook (see the "Example Usage" section).

#### 3. Functional Connectome Thresholder 
Available soon...

### Example Usage:
A simple usage example can be found in notebooks/example.ipynb <br>

### Documentation:
You can find the NiFeatures documentation following this [link here]() (not available, yet).
