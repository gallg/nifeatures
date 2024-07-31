from .transform import DisplacementInvariantTransformer
from sklearn.base import is_classifier, is_regressor
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import nibabel as nib
import pandas as pd
import numpy as np
import itertools
import joblib


def _update_parameters(parameters, settings):
    param_keys = list(parameters.keys())
    settings_keys = list(settings.keys())

    duplicates = [duplicate for duplicate in settings_keys
                  if duplicate in param_keys]

    if len(duplicates) > 0:
        for element in duplicates:
            del settings[element]

    parameters.update(settings)
    return parameters


def _get_search_params(x, y, p_grid, iteration=None):
    xy_hash = {"X": joblib.hash(str(x)), "y": joblib.hash(str(y))}

    # Get the search algorithm parameters for the current iteration;
    keys = [key.split("__")[1] for key in p_grid.keys()]
    values = list(itertools.product(*p_grid.values()))[iteration]
    params = dict(zip(keys, values))

    # Hash X, y and other parameters together;
    hash_params = joblib.hash((str(xy_hash), str(params)))

    return keys, values, params, hash_params


class TransformerCV:
    def __init__(
            self,
            estimator,
            p_grid,
            *,
            transform=False,
            mask=None,
            settings=None,
            scoring=None,
            cv=None,
            shuffle=False,
            random_state=None,
            n_jobs=None
    ):
        self.estimator = estimator
        self.parameters = p_grid
        self.transform = transform
        self.scorer = scoring

        # If settings is None, run the transformer with default parameters;
        if settings is None:
            self.settings = dict()
        else:
            self.settings = settings

        # load a mask in case of data transformation;
        if transform is True and mask is None:
            raise TypeError("A mask must be provided if transform is True.")
        elif isinstance(mask, str):
            self.mask = nib.load(mask)
        elif isinstance(mask, nib.Nifti1Image):
            self.mask = mask

        # set up a sklearn scorer;
        if scoring is None:
            if is_classifier(estimator):
                self.scorer = get_scorer("accuracy")
            elif is_regressor(estimator):
                self.scorer = get_scorer("r2")
        else:
            self.scorer = get_scorer(scoring)

        # set up cross validation strategy;
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
        elif isinstance(cv, object):
            self.cv = cv
            self.cv.shuffle = shuffle
            self.cv.random_state = random_state
        else:
            raise TypeError("The cv parameter must be an integer, a class or None.")

        # define number of jobs;
        if n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = n_jobs

        # define cache and result variables;
        self._cache = []
        self.precomputed = []
        self.models = []

    def _precompute(self, x, y, p_grid, iteration=None, train_index=None):

        x_train, y_train = x[train_index], y[train_index]

        keys, values, params, hash_params = _get_search_params(
            x_train,
            y_train,
            p_grid,
            iteration
        )

        if hash_params not in self._cache:
            # Add transformer settings to the current set of parameters, then precompute data;
            self._cache.append(hash_params)
            params = _update_parameters(params, self.settings)

            coords = DisplacementInvariantTransformer(**params).fit(
                x_train,
                y_train
            ).coords_
        else:
            coords = np.nan

        return keys, values, hash_params, coords

    def _transform_precomputed(self, x, y, iteration, train_index, test_index=None, mask=None):
        x_train, y_train = x[train_index], y[train_index]
        x_test, y_test = x[test_index], y[test_index]

        # infer estimator parameters from pre-computed data;
        keys = self.precomputed[iteration, 0]
        values = self.precomputed[iteration, 1]
        coords = self.precomputed[iteration, -1]
        params = dict(zip(keys, values))

        model_params = {key: value for key, value in zip(params.keys(), params.values())
                        if key in self.estimator().get_params().keys()}

        # transform precomputed coordinates;
        trf = DisplacementInvariantTransformer()
        trf.mask = mask
        trf.coords_ = coords
        x_trf = trf.transform(x_train)

        model = self.estimator(**model_params).fit(x_trf, y_train)
        y_pred = model.predict(trf.transform(x_test))
        score = self.scorer._score_func(y_test, y_pred)

        return model, score

    def fit(self, x, y, groups=None):

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        # n_iteration is equal to number of parameter combinations;
        n_iterations = len(list(itertools.product(*self.parameters.values())))

        # Run precomputation for every combination of parameters and cv-fold;
        self.precomputed.append(Parallel(n_jobs=self.n_jobs)(delayed(
            self._precompute)(x, y, self.parameters, iteration, train_index)
                for train_index, _ in self.cv.split(x, y, groups=groups)
                for iteration in np.arange(n_iterations)))

        # Remove possible NAs and return precomputed data as a numpy array;
        self.precomputed = pd.DataFrame.from_records(
            self.precomputed[0]).dropna().to_numpy()

        if self.transform is False:
            return self.precomputed
        else:
            # ToDo: check that train and test index always coincide with pre-computation;
            self.models.append(Parallel(n_jobs=self.n_jobs)(delayed(
                self._transform_precomputed)(x, y, iteration, train_index, test_index, self.mask)
                    for train_index, test_index in self.cv.split(x, y, groups=groups)
                    for iteration in np.arange(n_iterations)))
            return self
