__author__ = 'Nikolay Arefyev'

from math import ceil

import numpy as np

from .parallel import parallel_map


def argmaxk_rows_basic(arr, k=10, sort=False):
    """
    Reference non-optimized implementation.
    """
    if sort:
        return np.argsort(arr, axis=1)[:, :-k - 1:-1]
    else:
        return np.argpartition(arr, kth=-k, axis=1)[:, -k:]


def argmaxk_rows_opt1(arr, k=10, sort=False):
    """
    Optimized implementation. When sort=False it is equal to argmaxk_rows_basic. When sort=True and k << arr.shape[1],
    it is should be faster, because we argsort only subarray of k max elements from each row of arr (arr.shape[0] x k) instead of
    the whole array arr (arr.shape[0] x arr.shape[1]).
    """
    best_inds = np.argpartition(arr, kth=-k, axis=1)[:, -k:]  # column indices of k max elements in each row (m x k)
    if not sort:
        return best_inds
    # generate row indices corresponding to best_ids (just current row id in each row) (m x k)
    rows = np.arange(best_inds.shape[0], dtype=np.intp)[:, np.newaxis].repeat(best_inds.shape[1], axis=1)
    best_elems = arr[rows, best_inds]  # select k max elements from each row using advanced indexing (m x k)
    # indices which sort each row of best_elems in descending order (m x k)
    best_elems_inds = np.argsort(best_elems, axis=1)[:, ::-1]
    # reorder best_indices so that arr[i, sorted_best_inds[i,:]] will be sorted in descending order
    sorted_best_inds = best_inds[rows, best_elems_inds]
    return sorted_best_inds


def argmaxk_rows(arr, k=10, sort=False, impl='opt1', nthreads=8):
    """
    Returns column indices of k max elements in each row of input ndarray.
    :param arr: input ndarray (m x n)
    :param k: number of maximum values to find in each row of input matrix
    :param sort: for each row of input matrix returned column indices should sort k max elements of this row in
    descending order (particularly, res[:,0] equals to argmax(arr, axis=1) )
    :param impl: single thread implementation to use (look at implementations docs)
    :param nthreads: split input array into this many parts and process in parallel; takes advantage of the fact that
    numpy.argpartition is not parallelized
    :return: output ndarray (m x k) of column indices of k max elements in each row of input matrix
    """
    IMPLS = {'basic': argmaxk_rows_basic, 'opt1': argmaxk_rows_opt1}
    if impl not in IMPLS:
        raise ValueError('Unknown value of parameter: impl=%s; possible values: %r' % (impl, IMPLS.keys()))
    fimpl = IMPLS[impl]

    if nthreads == 1:
        res = fimpl(arr, k, sort)
    else:
        m = arr.shape[0]
        batchsize = int(ceil(1. * m / nthreads))

        def ppp(i):
            return fimpl(arr[i:i + batchsize, :], k, sort)

        lres = parallel_map(ppp, range(0, m, batchsize), threads=nthreads)
        res = np.vstack(lres)

    return res
