__author__ = 'Nikolay Arefyev'

import numpy as np
import psutil

from .argmaxk import argmaxk_rows


def nn_vec_basic(arr1, arr2, topn, sort=True, return_sims=False, nthreads=8):
    """
    For each row in arr1 (m1 x d) find topn most similar rows from arr2 (m2 x d). Similarity is defined as dot product.
    Please note, that in the case of normalized rows in arr1 and arr2 dot product will be equal to cosine and will be
    monotonically decreasing function of Eualidean distance.
    :param arr1: array of vectors to find nearest neighbours for
    :param arr2: array of vectors to search for nearest neighbours in
    :param topn: number of nearest neighbours
    :param sort: indices in i-th row of returned array should sort corresponding rows of arr2 in descending order of
    similarity to i-th row of arr2
    :param return_sims: return similarities along with indices of nearest neighbours
    :param nthreads:
    :return: array (m1 x topn) where i-th row contains indices of rows in arr2 most similar to i-th row of m1, and, if
    return_sims=True, an array (m1 x topn) of corresponding similarities.
    """
    sims = np.dot(arr1, arr2.T)
    best_inds = argmaxk_rows(sims, topn, sort=sort, nthreads=nthreads)
    if not return_sims:
        return best_inds

    # generate row indices corresponding to best_inds (just current row id in each row) (m x k)
    rows = np.arange(best_inds.shape[0], dtype=np.intp)[:, np.newaxis].repeat(best_inds.shape[1], axis=1)
    return best_inds, sims[rows, best_inds]


def nn_vec(m1, m2, topn, sort=True, return_sims=False, nthreads=8, USE_MEM_PERCENT=0.3, verbose=True):
    ndists = m1.shape[0] * m2.shape[0]  # number of distances

    if m1.shape[0] < 2 or ndists < 10 ** 7:  # cannot or need not split m1 into batches
        return nn_vec_basic(m1, m2, topn=topn, sort=sort, return_sims=return_sims, nthreads=nthreads)

    # estimate memory required to store results:
    # best_inds: m1.shape[0] * topn * tmp1.itemsize, dists: m1.shape[0] * topn * tmp2.itemsize
    tmp_inds, tmp_dists = nn_vec_basic(m1[:2, :], m2[:2, :], topn=2, sort=False, return_sims=True, nthreads=1)
    res_mem = m1.shape[0] * topn * (tmp_inds.itemsize + (tmp_dists.itemsize if return_sims else 0))

    amem = psutil.virtual_memory().available
    use_mem = (amem - res_mem) * USE_MEM_PERCENT
    dists_mem = ndists * tmp_dists.itemsize  # memory required for the whole distances matrix
    num_batches = int(np.ceil(dists_mem / use_mem))
    batch_size = int(np.ceil(1.0 * m1.shape[0] / num_batches))
    if verbose:
        print(
            'Full distances matrix will occupy %.2fG; we would like to occupy %.2fG from %.2fG of available memory...' % \
            (1. * dists_mem / 2 ** 30, 1. * use_mem / 2 ** 30, 1. * amem / 2 ** 30))
        print('... processing in %d batches of %d rows' % (num_batches, batch_size))

    res_inds, res_dists = None, None
    for st in range(0, m1.shape[0], batch_size):
        en = st + batch_size
        if verbose:
            print('Processing rows %d-%d from %d' % (st, min(en - 1, m1.shape[0]), m1.shape[0]))
        res = nn_vec_basic(m1[st:en, :], m2, topn=topn, sort=sort, return_sims=return_sims, nthreads=nthreads)
        res0 = res[0] if return_sims else res
        res_inds = np.vstack([res_inds, res0]) if res_inds is not None else res0
        if return_sims:
            res_dists = np.vstack([res_dists, res[1]]) if res_dists is not None else res[1]
    return (res_inds, res_dists) if return_sims else res_inds
