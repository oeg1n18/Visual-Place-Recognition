import numpy as np
from scipy.stats import norm
from typing import Union


def best_match_per_query(S: np.ndarray) -> np.ndarray:
    """
    Finds the best match per query from a similarity matrix S.

    Args:
        S (np.ndarray): A two-dimensional similarity matrix with continuous values.
            Higher values indicate higher similarity.

    Returns:
        np.ndarray: A two-dimensional boolean matrix with the same shape as S,
            where the best match per query is marked as True.
    """
    i = np.argmax(S, axis=0)
    j = np.int64(range(len(i)))

    M = np.zeros_like(S, dtype='bool')
    M[i, j] = True

    return M


def thresholding(S: np.ndarray, thresh: Union[str, float]) -> np.ndarray:
    """
    Applies thresholding on a similarity matrix S based on a given threshold or
    an automatic thresholding method.

    The automatic thresholding is based on Eq. 2-4 in Schubert, S., Neubert, P. &
    Protzel, P. (2021). Beyond ANN: Exploiting Structural Knowledge for Efficient
    Place Recognition. In Proc. of Intl. Conf. on Robotics and Automation (ICRA).
    DOI: 10.1109/ICRA48506.2021.9561006

    Args:
        S (np.ndarray): A two-dimensional similarity matrix with continuous values.
            Higher values indicate higher similarity.
        thresh (Union[str, float]): A threshold value or the string 'auto' to apply
            automatic thresholding.

    Returns:
        np.ndarray: A two-dimensional boolean matrix with the same shape as S,
            where values greater or equal to the threshold are marked as True.
    """
    if thresh == 'auto':
        mu = np.median(S)
        sig = np.median(np.abs(S - mu)) / 0.675
        thresh = norm.ppf(1 - 1e-6, loc=mu, scale=sig)

    M = S >= thresh

    return M