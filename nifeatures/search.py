from .transform import DisplacementInvariantTransformer
from sklearn.base import is_classifier, is_regressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import nibabel as nib
import pandas as pd
import numpy as np
import itertools


def r2_score(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]**2


def _get_search_params(p_grid, iteration=None):
    keys = list(p_grid.keys())
    values = list(itertools.product(*p_grid.values()))[iteration]
    params = dict(zip(keys, values))
    return keys, values, params


def _split_params(params):
    trf_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith('trf__')}
    model_params = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith('model__')}
    return trf_params, model_params


class DisplacementInvariantTransformerCV:
    def __init__(
            self,
            estimator,
            p_grid,
            *,
            mask=None,
            scoring=None,
            cv=None,
            shuffle=False,
            refit=True,
            random_state=None,
            n_jobs=None
    ):
        """Hyperparameter search for models that include an instance of 
        DisplacementInvariantTransformer.

        Unlike GridSearchCV, DisplacementInvariantTransformerCV does not need to take in
        a scikit-learn pipeline that includes DisplacementInvariantTransformer. Instead, it
        takes in a scikit-learn estimator and a dictionary of hyperparameter ranges.

        the parameter grid is a scikit-learn compatible dictionary of parameters.

        Args:
            estimator (callable): 
                A scikit-learn estimator interface passed as a callable. 
                It doesn't need to contain DisplacementInvariantTransformer.
            p_grid (dict): 
                Dictionary with parameter names as keys and lists of parameter 
                settings to try as values.
            mask (str, Nifti1Image optional): 
                Brain mask used to perform calculations on the brain space. 
                If None, the MNI152 2mm template is used. Defaults to None.
            scoring (callable, optional): 
                Strategy to evaluate the performance of the cross-validated 
                model on the test set. If None, scoring defaults to "accuracy_score"
                or "r2_score" for classification and regression tasks respectively. 
                Defaults to None.
            cv (int, cross-validation generator, optional): 
                Determines the cross-validation splitting strategy. 
                If None, use the default 5-fold cross-validation. Defaults to None.
            shuffle (bool, optional): 
                If True, shuffle the data before splitting into batches. 
                The samples within each split will not be shuffled. Defaults to False.
            refit (bool, optional): 
                If True, returns a fitted version of pipeline as best_model_. Defaults to True.
            random_state (int, RandomState instance, optional): 
                When shuffle is True, random_state determines randomization of each fold. 
                Defaults to None.
            n_jobs (int, optional): 
                Number of jobs to run in parallel. If None, n_jobs is set to 1.
                set it to -1 to use all CPU cores. Defaults to None.

        Attributes:
            coordinates_ (pandas DataFrame): 
                The parameters, score and coordinates computed by the transformer
                for each model evaluated using cross-validation.
            best_model_ (scikit-learn Pipeline): 
                A pipeline that includes the best model found using 
                DisplacementInvariantTransformer and the specified estimator.
                If refit is set to False, the best_model_ is not fitted.
        """

        self.estimator = estimator
        self.parameters = p_grid
        self.scorer = scoring
        self.refit = refit

        if mask is None:
            self.mask = mask
        elif isinstance(mask, str):
            self.mask = nib.load(mask)
        elif isinstance(mask, nib.Nifti1Image):
            self.mask = mask
        else:
            raise TypeError("The mask must be a filepath string or a Nifti1Image object.")

        if scoring is None:
            if is_classifier(estimator):
                self.scorer = accuracy_score
            elif is_regressor(estimator):
                self.scorer = r2_score
            if callable(scoring):
                self.scorer = scoring

        if cv is None:
            self.cv = KFold(
                n_splits=5,
                random_state=random_state,
                shuffle=shuffle
            )
        elif isinstance(cv, int):
            self.cv = KFold(
                n_splits=cv,
                random_state=random_state,
                shuffle=shuffle
            )
        elif hasattr(cv, 'split'):
            self.cv = cv
            self.cv.shuffle = shuffle
            self.cv.random_state = random_state
        else:
            raise TypeError("The cv parameter must be an integer, a cross-validation instance or None.")

        self.n_jobs = 1 if n_jobs is None else n_jobs
        self.best_model_ = None
        self.coordinates_ = []

    def _compute_models(
        self, 
        x, 
        y, 
        p_grid, 
        iteration=None, 
        train_index=None, 
        test_index=None
    ):
        keys, values, params = _get_search_params(
            p_grid,
            iteration
        )

        trf_params, model_params = _split_params(params)

        transformer, model, score = self._fit_model(
            x, 
            y, 
            train_index,
            test_index,
            trf_params, 
            model_params, 
            predict=True
        )

        return keys, values, score, transformer.coords_, (train_index, test_index)

    def _fit_model(
        self, 
        x, 
        y, 
        train_index,
        test_index,
        trf_params, 
        model_params, 
        predict=False
    ):
        score = None

        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        transformer = DisplacementInvariantTransformer(**trf_params, mask=self.mask).fit(x_train, y_train)
        x_trf = transformer.transform(x_train)
        
        model = self.estimator(**model_params).fit(x_trf, y_train)

        if predict:
            y_pred = model.predict(transformer.transform(x_test))
            score = self.scorer(y_test, y_pred)
        
        return transformer, model, score

    
    def _return_best_model(self, x, y):
        best_index = np.argmax(self.coordinates_["score"])
        keys = self.coordinates_["keys"][best_index]
        values = self.coordinates_["values"][best_index]
        params = dict(zip(keys, values))

        trf_params, model_params = _split_params(params)

        if self.refit:
            train_index = self.coordinates_["indices"][best_index][0]
            test_index = self.coordinates_["indices"][best_index][1]
            
            transformer, model, score = self._fit_model(
                x, 
                y, 
                train_index,
                test_index,
                trf_params, 
                model_params, 
                predict=False
            )
        else:
            transformer = DisplacementInvariantTransformer(**trf_params, mask=self.mask)
            model = self.estimator(**model_params)

        pipeline = Pipeline([
            ('trf', transformer),
            ('model', model)
        ])

        return pipeline

    def fit(self, x, y, groups=None):
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        n_iterations = len(list(itertools.product(*self.parameters.values())))

        results = (Parallel(n_jobs=self.n_jobs)(delayed(
            self._compute_models)(x, y, self.parameters, iteration, train_index, test_index)
            for train_index, test_index in self.cv.split(x, y, groups=groups)
            for iteration in np.arange(n_iterations)))

        records = [
            {
                'keys': result[0],
                'values': result[1],
                'score': result[2],
                'coordinates': result[3],
                'indices': result[4]
            }
            for result in results
        ]

        self.coordinates_ = pd.DataFrame.from_records(records)
        self.best_model_ = self._return_best_model(x, y)

        return self
