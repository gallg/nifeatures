from .transform import DisplacementInvariantTransformer
from sklearn.base import is_classifier, is_regressor
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
    keys = [key.split("__")[1] for key in p_grid.keys()]
    values = list(itertools.product(*p_grid.values()))[iteration]
    params = dict(zip(keys, values))
    return keys, values, params


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
            random_state=None,
            n_jobs=None
    ):
        self.estimator = estimator
        self.parameters = p_grid
        self.scorer = scoring

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
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.coordinates_ = []

    def _compute_models(self, x, y, p_grid, iteration=None, train_index=None, test_index=None, mask=None):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        keys, values, params = _get_search_params(
            p_grid,
            iteration
        )

        transformer = DisplacementInvariantTransformer(**params, mask=mask).fit(x_train, y_train)
        x_trf = transformer.transform(x_train)

        model_params = {key: value for key, value in zip(params.keys(), params.values())
                        if key in self.estimator().get_params().keys()}

        model = self.estimator(**model_params).fit(x_trf, y_train)
        y_pred = model.predict(transformer.transform(x_test))
        score = self.scorer(y_test, y_pred)

        # ToDo: fix best results;
        if score > self.best_score_:
            self.best_score_ = score
            self.best_model_ = model
            self.best_params_ = dict(zip(keys, values))

        return keys, values, score, transformer.coords_

    def fit(self, x, y, groups=None):
        if not isinstance(y, np.ndarray):
            y = np.array(y)

        n_iterations = len(list(itertools.product(*self.parameters.values())))

        # ToDo: return best model with its transformer in a pipeline;
        self.coordinates_.append(Parallel(n_jobs=self.n_jobs)(delayed(
            self._compute_models)(x, y, self.parameters, iteration, train_index, test_index, self.mask)
            for train_index, test_index in self.cv.split(x, y, groups=groups)
            for iteration in np.arange(n_iterations)))

        self.coordinates_ = pd.DataFrame(
            columns=[
                'keys',
                'values',
                'score',
                'coordinates'
            ]
        ).from_records(self.coordinates_[0])

        return self
