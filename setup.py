from setuptools import setup, find_packages

VERSION = '0.1.0'
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
    install_requires=["joblib==1.4.2", "nibabel==5.2.1", "nilearn==0.10.4", "numpy==1.26.4",
                      "pandas==2.2.2", "scikit_learn==1.5.1", "scipy==1.13.1"],
    keywords=['python', 'machine learning', 'neuroimaging', 'feature engineering'],
    classifiers=[]
)
