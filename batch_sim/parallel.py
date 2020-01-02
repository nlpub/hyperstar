__author__ = 'Nikolay Arefyev'

import sys
import threading
from itertools import count


def foreach(f, l, threads=3, return_=False):
    """
    Apply f to each element of l, in parallel. Return list [f(v) for v in l], the order of results f(v) is the same as
    the order of inputs v in l.
    """

    if threads > 1:
        iteratorlock = threading.Lock()
        exceptions = []
        if return_:
            n = 0
            d = {}
            i = zip(count(), l.__iter__()).__iter__()
        else:
            i = l.__iter__()

        def runall():
            while True:
                iteratorlock.acquire()
                try:
                    try:
                        if exceptions:
                            return
                        v = i.__next__()
                    finally:
                        iteratorlock.release()
                except StopIteration:
                    return
                try:
                    if return_:
                        n, x = v
                        d[n] = f(x)
                    else:
                        f(v)
                except:
                    e = sys.exc_info()
                    iteratorlock.acquire()
                    try:
                        exceptions.append(e)
                    finally:
                        iteratorlock.release()

        threadlist = [threading.Thread(target=runall) for j in range(threads)]
        for t in threadlist:
            t.start()
        for t in threadlist:
            t.join()
        if exceptions:
            a, b, c = exceptions[0]
            raise a(b).with_traceback(c)
        if return_:
            r = sorted(d.items(), key=lambda t: t[0])  # the results should be in the same order as inputs in l
            return [v for (n, v) in r]
    else:
        if return_:
            return [f(v) for v in l]
        else:
            for v in l:
                f(v)
            return


def parallel_map(f, l, threads=3):
    return foreach(f, l, threads=threads, return_=True)
