from scipy.special import softmax
from scipy.signal import argrelextrema
from warnings import warn
import numpy as np


def calculate_r2(x, y):
    return pairwise_correlation(x, y) ** 2


def pairwise_correlation(a, b):
    am = a - np.mean(a, axis=0, keepdims=True)
    bm = b - np.mean(b, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(
        np.sum(
            am**2, 
            axis=0,
            keepdims=True
        )).T * np.sqrt(

        np.sum(
            bm**2, 
            axis=0, 
            keepdims=True
        )))


def check_distances(coord, x_indices, y_indices, z_indices, min_distance):
    distances = np.sqrt((x_indices - coord[0]) ** 2
                        + (y_indices - coord[1]) ** 2
                        + (z_indices - coord[2]) ** 2)
    return distances >= min_distance


def filter_coords(coords, min_distance):
    selected_coords = []

    while len(coords) > 0:
        current_coord = coords[0]
        x_indices, y_indices, z_indices = coords.T

        mask = check_distances(
            current_coord, 
            x_indices, 
            y_indices, 
            z_indices, 
            min_distance
        )

        selected_coords.append(current_coord)
        coords = coords[mask]

    return np.array(selected_coords)


def draw_sphere(empty_map, mask, coord, radius):
    x_indices, y_indices, z_indices = np.indices(empty_map.shape)
    distances = np.sqrt((x_indices - coord[0]) ** 2
                        + (y_indices - coord[1]) ** 2
                        + (z_indices - coord[2]) ** 2)

    sphere = distances <= radius
    sphere = np.logical_and(sphere, mask)

    return sphere


def update_stats_map(stats, temperature, thr=0):

    # Skip thresholding if threshold is already equal to 0:
    if thr != 0:
        if thr >= stats.max():
            thr = 0
            warn(("The selected threshold is higher than the max value in " +
                  "the statistical map. Using a threshold of 0."))

    # Threshold data, apply temperature and calculate softmax;
    stats[np.abs(stats) <= thr] = 0
    stats /= temperature
    stats[stats != 0] = softmax(stats[stats != 0])

    return stats


def calculate_peaks(coords, n_peaks, min_distance, verbose):

    # Keep coords that respect min_distance;  
    coords = filter_coords(coords, min_distance)

    # Make sure to keep only the best "n_peaks" coordinates;
    if verbose and len(coords) < n_peaks:
        warn("Maximum number of eligible local maxima is lower then n_peaks. "
                    + "Using n_peaks == {}".format(coords.shape[0]))

    return coords


def get_peak_probabilities(
        stats_map,
        n_peaks,
        min_distance, 
        temperature, 
        thr,
        tol, 
        random_state, 
        verbose
    ):

    map_length = np.flatnonzero(stats_map).shape[0]

    # Apply temperature and get softmax probabilities;
    probability_map = update_stats_map(
        stats_map.copy(),
        temperature,
        thr=thr
    )

    use_max_values = ((1-tol) <= np.max(probability_map) <= (1+tol))

    if use_max_values:
        # If temperature is too low, use max values to define peaks;
        values = np.abs(stats_map.flatten())
        coords = np.argsort(values)[::-1][:map_length]
    else:
        # Otherwise return coordinates sampled using softmax probability;
        rng = np.random.default_rng(random_state)
        random_nums = rng.uniform(size=map_length)
        coords = np.searchsorted(np.cumsum(probability_map), random_nums)

    result = np.apply_along_axis(
        np.unravel_index, 
        0, 
        coords, 
        stats_map.shape
    ).T

    # make sure that coordinates are within the number of peaks;
    result = np.unique(result, axis=0)[:n_peaks, :]

    # Get coordinates that respect min_distance;
    if min_distance > 0:
        result = calculate_peaks(result, n_peaks, min_distance, verbose)

    return result


def find_peaks(
        stats_map,
        mask,
        n_peaks=100,
        min_distance=2,
        probability=None,
        temperature=0.1,
        thr=0,
        tol=1e-4,
        random_state=None,
        verbose=False
    ):

    coords = []
    stats_map = stats_map.get_fdata() * mask.get_fdata().astype(bool)

    # Use softmax to find random peaks in the data;
    if probability == 'softmax':
        coords = get_peak_probabilities(stats_map,
                                        n_peaks,
                                        min_distance,
                                        temperature,
                                        thr=thr,
                                        tol=tol,
                                        random_state=random_state,
                                        verbose=verbose
                                    )

    elif probability is None:
        peaks0 = np.array(
            argrelextrema(stats_map, np.greater, axis=0, order=1))
        peaks1 = np.array(
            argrelextrema(stats_map, np.greater, axis=1, order=1))
        peaks2 = np.array(
            argrelextrema(stats_map, np.greater, axis=2, order=1))

        stacked = np.vstack((peaks0.transpose(),
                             peaks1.transpose(),
                             peaks2.transpose())
                            )

        # Keep coordinates that appear three times (once for each axis);
        elements, counts = np.unique(stacked, axis=0, return_counts=True)
        coords = elements[np.where(counts == 3)[0]]

        # Get "n_peaks" coordinates sorted for highest statistical value;
        # make sure that coordinates are within the number of peaks;
        values = stats_map[coords[:, 0], coords[:, 1], coords[:, 2]]
        coords = coords[values.argsort()[::-1]][:n_peaks, :]
        
        if min_distance > 0:
            coords = calculate_peaks(coords, n_peaks, min_distance, verbose)

    return coords
