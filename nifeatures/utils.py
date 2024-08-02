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
        np.sum(am**2, axis=0,
               keepdims=True)).T * np.sqrt(
        np.sum(bm**2, axis=0, keepdims=True)))


def draw_sphere(empty_map, mask, coord, radius):

    # Calculate Euclidean distance;
    x_indices, y_indices, z_indices = np.indices(empty_map.shape)
    distances = np.sqrt((x_indices - coord[0]) ** 2
                        + (y_indices - coord[1]) ** 2
                        + (z_indices - coord[2]) ** 2)

    # Use the mask to make sure the sphere is not drawn out of the brain;
    sphere = distances <= radius
    sphere = np.logical_and(sphere, mask)
    empty_map[sphere] = 1

    return empty_map


def create_roi(empty_map, mask, coord, radius, flatten=False):

    # Apply the maximum filter and get roi indices;
    filtered_arr = draw_sphere(empty_map, mask, coord, radius)

    # Get indices, then restore the empty map to avoid making copies;
    if flatten is True:
        indices = np.flatnonzero(filtered_arr)
        empty_map[np.unravel_index(indices, empty_map.shape)] = 0
    else:
        indices = np.nonzero(filtered_arr)
        empty_map[indices] = 0

    return indices


def reduce_coords(stats, empty_map, coords, n_peaks, min_distance):

    result = []

    for coord in coords:
        x, y, z = coord

        if not stats[(x, y, z)]:
            continue

        roi = create_roi(empty_map, stats, coord, min_distance)
        stats[roi] = 0

        result.append(coord)

        if len(result) == n_peaks:
            break

    return result


def update_stats_map(stats, temperature=None, use_softmax=False, thr=0):

    # Skip thresholding if threshold is already equal to 0:
    if thr != 0:
        if thr >= stats.max():
            thr = 0
            warn(("The selected threshold is higher than the max value in " +
                  "the statistical map. Using a threshold of 0."))

    # Threshold data;
    stats[stats <= thr] = 0

    # Apply temperature if needed;
    if temperature is not None:
        stats /= temperature

    # Calculate softmax on the current statistical map;
    if use_softmax:
        stats[stats != 0] = softmax(stats[stats != 0])

    return stats


def get_peak_probabilities(stats_map, empty_map, n_peaks,
                           min_distance, temperature, thr,
                           tol, random_state):

    if empty_map is None:
        empty_map = np.zeros(stats_map.shape)

    result = []
    map_length = np.flatnonzero(stats_map).shape[0]

    # Apply temperature and get softmax probabilities;
    probability_map = update_stats_map(
        stats_map.copy(),
        temperature=temperature,
        use_softmax=True,
        thr=thr
    )

    # If temperature is too low, use max values to define peaks;
    use_max_values = ((1-tol) <= np.max(probability_map) <= (1+tol))

    if use_max_values:
        coords = np.argsort(abs(stats_map.flatten()))[::-1][:map_length]
    else:
        rng = np.random.default_rng(random_state)
        random_nums = rng.uniform(size=map_length)
        coords = np.searchsorted(np.cumsum(probability_map), random_nums)

    for coord in coords:
        result.append(np.unravel_index(coord, stats_map.shape))

    # Get coordinates that respect min_distance;
    if min_distance == 0:
        result = np.array(result[:n_peaks])

    elif min_distance > 0:
        result = reduce_coords(stats_map.copy(),
                               empty_map,
                               result,
                               n_peaks,
                               min_distance
                               )

        if len(result) < n_peaks:
            warn("The number of coordinates that respect min_distance "
                 + "is lower than n_peaks. "
                 + "Using n_peaks == {}".format(len(result)))

    else:
        raise ValueError(
            "{} is an invalid value for min_distance".format(min_distance)
            )

    return np.array(result)


def calculate_peaks(coords, n_peaks, min_distance):

    result = []
    for peak in range(n_peaks):

        if peak > (coords.shape[0]-1):
            warn("Maximum number of local maxima is lower then n_peaks. "
                 + "Using n_peaks == {}".format(len(result)))
            break

        result.append(coords[peak])
        x0, y0, z0 = coords[peak]
        dist = []

        for coord in range(coords.shape[0]):
            x, y, z = coords[coord]
            distance = np.sqrt(((x - x0) ** 2 + (y - y0) ** 2 + (z - z0)**2))
            dist.append(distance)

        coords = coords[np.array(dist) >= min_distance]

    return np.array(result)


def find_peaks(
        stats_map,
        mask,
        empty_map=None,
        n_peaks=100,
        min_distance=2,
        probability=None,
        temperature=0.1,
        thr=0,
        tol=1e-4,
        random_state=None
):

    coords = []

    stat_mask = mask.get_fdata()
    stats_map = stats_map.get_fdata()
    stats_map[stat_mask.astype(bool) == 0] = 0

    # Use softmax to find random peaks in the data;
    if probability == 'softmax':
        coords = get_peak_probabilities(stats_map,
                                        empty_map,
                                        n_peaks,
                                        min_distance,
                                        temperature,
                                        thr=thr,
                                        tol=tol,
                                        random_state=random_state
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

        # Get coordinates sorted for highest value with length == n_peaks;
        # Use the maximum number of peaks if n_peaks > len(coords);
        values = stats_map[coords[:, 0], coords[:, 1], coords[:, 2]]
        coords = coords[values.argsort()[::-1]]
        coords = calculate_peaks(coords, n_peaks, min_distance)

    return coords
