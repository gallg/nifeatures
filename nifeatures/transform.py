from sklearn.base import BaseEstimator, TransformerMixin
from nilearn.datasets import load_mni152_brain_mask
from sklearn.utils.validation import check_is_fitted
from nifeatures.utils import _create_roi, _calculate_r2, find_peaks
from collections import abc
import nibabel as nib
import pandas as pd
import numpy as np
import joblib


class DisplacementInvariantTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, *, mask=None, n_peaks=100, radius=4, min_distance=2,
                 method='R2', proba=None, temperature=0.2, threshold=0,
                 tolerance=1e-4, aggregator_func=[np.max], precomp=None,
                 random_state=None, **kwargs):

        """ Transforms brain data to generate new, more important and
            concise features.

            The program finds voxels (peaks) in X that are highly related to
            the target y and generates spheres with specific radius using
            those voxels as sphere centers. The voxels belonging to each
            sphere are then used to generate aggregated measures through one
            or more user specified aggregator functions.

        Args:
            mask (str, optional):
                Path to Binary template mask.
                If None, the MNI152 2mm template is used. Defaults to None.
            n_peaks (int, optional):
                Number of signal peaks to find. Defaults to 100.
            radius (int, optional):
                Radius of the spheres generated around each peak.
                Defaults to 4.
            min_distance (int, optional):
                Minimum distance between peaks in voxels. Defaults to 2.
            method (str, callable, optional):
                Statistical function used to find peaks in the data.
                A custom function is also accepted. Defaults to 'R2'.
            proba (str, optional):
                If set to 'softmax' it generates spheres using
                softmax probability values. Defaults to None.
            temperature (float, optional):
                If proba is equal to 'softmax', determines whether the peaks
                are sampled from higher or lower probability values.
                Defaults to 0.2.
            threshold (float, optional):
                If proba is equal to 'softmax', every value in the statistical
                map lower then the threshold is not considered during
                softmax calculations. Defaults to 0.
            tolerance (float, optional):
                If proba is equal to 'softmax' and temperature is very low,
                select peaks in a quasi-deterministic way by using
                maximum values. Defaults to 1e-4.
            aggregator_func (list, optional):
                List of functions used to generate the new features.
                Defaults to [np.max].
            precomp (array-like, optional):
                None or array with precomputed hashes and coordinates to get a
                faster parameters search. Defaults to None.
            random_state (int, optional):
                If proba is equal to 'softmax', set seed to make sphere
                generation reproducible. Defaults to None.

        Returns:
            X_out (np.ndarray): Array containing new features with shape:
            n_participants x (n_ROIs * n_funcs).

        """

        if isinstance(precomp, pd.DataFrame):
            self.precomp = precomp.to_numpy()
        else:
            self.precomp = precomp

        self.mask = mask
        self.n_peaks = n_peaks
        self.radius = radius
        self.min_distance = min_distance
        self.method = method
        self.aggregator_func = aggregator_func
        self.proba = proba
        self.temperature = temperature
        self.threshold = threshold
        self.tolerance = tolerance
        self.random_state = random_state

    def get_precomputed_data(self, X, y):

        # Get hash for the current X and y;
        Xy_hash = {"X": joblib.hash(str(X)), "y": joblib.hash(str(y))}

        # Get current parameters used by the search algorithm;
        search_params = self.precomp[0][0]
        current_params = tuple(
            [self.__getattribute__(search_params[parameter])
                for parameter in range(len(search_params))]
                    )
        current_params = dict(zip(search_params, current_params))

        # Hash X, y and all the parameters together;
        hash_params = joblib.hash((
            str(Xy_hash),
            str(current_params)
            ))
        current_hash = [hash_params == self.precomp[:, 2][hash]
                        for hash in range(len(self.precomp[:, 2]))]

        # For debug purposes only!!!
        # print(hash_params, current_params, ": ", True in current_hash)

        # If GridSearch "refit" parameter is True,
        # call fit normally to fit the best_estimator_;
        try:
            coords = self.precomp[:, -1][current_hash][0]
        except Exception:
            self.precomp = None
            coords = self.fit(X, y).coords_

        return coords

    def fit(self, X, y):

        if not isinstance(y, np.ndarray):
            y = np.array(y)

        if isinstance(self.mask, str):
            self.mask = nib.load(self.mask)
        elif self.mask is None:
            self.mask = load_mni152_brain_mask(resolution=2)
        elif isinstance(self.mask, nib.Nifti1Image):
            pass

        # Initialize empty_map to avoid re-allocation;
        self.empty_map = np.zeros(self.mask.get_data().shape)

        if self.precomp is not None:
            self.coords_ = self.get_precomputed_data(X, y)

        else:
            # Initialize array to contain group-level statistics;
            self.group_stats = np.zeros((X.shape[1], 2))

            # Select only non-zero std voxels for correlation;
            non_zero_mask = self.mask.get_fdata().flatten().astype(bool)
            non_zero_features = np.array(range(X.shape[1]))[non_zero_mask]
            non_zero_features = non_zero_features[
                (np.std(X[:, non_zero_mask],
                 axis=0) != 0)
                ]

            # Select correlation method and get group-level statistics;
            if self.method == 'R2':

                for feature in non_zero_features:
                    self.group_stats[feature, 0] = _calculate_r2(
                        X[:, feature],
                        y
                        )

            elif isinstance(self.method, abc.Callable):

                for feature in non_zero_features:
                    self.group_stats[feature, 0] = self.method(
                        X[:, feature],
                        y
                        )

            # Reshape group-level statistical map and get local maxima;
            self.group_map_ = nib.Nifti1Image(
                self.group_stats[:, 0].reshape(self.mask.get_fdata().shape),
                affine=self.mask.affine
                )
            self.coords_ = find_peaks(self.group_map_,
                                      mask=self.mask,
                                      empty_map=self.empty_map,
                                      n_peaks=self.n_peaks,
                                      min_distance=self.min_distance,
                                      proba=self.proba,
                                      temperature=self.temperature,
                                      thr=self.threshold,
                                      tol=self.tolerance,
                                      random_state=self.random_state
                                      )

        return self

    def transform(self, X):

        # Check if the Transformer is fitted;
        check_is_fitted(self)

        # Initialize the output array;
        n_samples = X.shape[0]
        index_mask = np.flatnonzero(self.mask.get_fdata())
        self.X_out_ = np.zeros((n_samples,))

        # Loop over each coordinate,
        # compute the corresponding sphere and aggregated values;
        for idx, coordinate in enumerate(self.coords_):
            indices = _create_roi(
                self.empty_map,
                coordinate,
                self.radius,
                flatten=True
            )
            indices = [idx for idx in indices if idx in index_mask]

            # Update temporary data;
            temp = self.aggregator_func[0](X[:, indices], axis=1)
            self.update_features(temp, n_samples, idx)

        return self.X_out_

    def update_features(self, temp, n_samples, idx):
        if len(temp.shape) > 1:
            temp = temp.T if temp.shape[0] > 1 else temp.reshape(n_samples,)

        if idx == 0:
            self.X_out_ = temp
        else:
            self.X_out_ = np.column_stack((self.X_out_, temp))

    def precompute(self, X, y):
        this = self.fit(X, y)
        return this.coords_

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)
