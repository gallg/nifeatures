from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'NiFeatures'
LONG_DESCRIPTION = 'A collection of scikit-learn compatible transformers for machine learning with MRI data'

# Setting up
setup(
    name="nifeatures",
    version=VERSION,
    author="PNI Lab (Predictive NeuroImaging Laboratory)",
    author_email="<giuseppe.gallitto@uk-essen.de>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["joblib==1.2.0", "nibabel==4.0.2", "nilearn==0.9.2", "numba==0.56.3", "numpy==1.23.4",
                      "pandas==1.5.1", "scikit_learn==1.2.1", "scipy==1.11.2"],
    keywords=['python', 'machine learning', 'neuroimaging', 'feature engineering'],
    classifiers=[]
)
