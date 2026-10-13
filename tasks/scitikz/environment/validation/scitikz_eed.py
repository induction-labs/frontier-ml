"""Accelerated Extended Edit Distance with the released parameters and float64 arithmetic.

The recurrence is the EED algorithm (Stanchev et al.); this implementation is
checked against torchmetrics' reference on randomized and real TeX strings.
"""


def compile_eed():
    import numpy as np
    from numba import njit
    @njit(nogil=True, fastmath=False)
    def recurrence(hyp, ref):
        length = len(hyp)
        previous = np.ones(length+1, dtype=np.float64)
        previous[0] = 0.0
        current = np.empty(length+1, dtype=np.float64)
        visits = np.full(length+1, -1, dtype=np.int64)
        for character in ref:
            current[0] = previous[0]+1.0
            cheapest, position = current[0], 0
            for j in range(1, length+1):
                substitution = previous[j-1] + (0.0 if hyp[j-1] == character else 1.0)
                value = min(current[j-1]+.2, substitution, previous[j]+1.0)
                current[j] = value
                if value < cheapest:
                    cheapest, position = value, j
            visits[position] += 1
            if character == 32:
                for j in range(length+1):
                    current[j] = min(current[j], cheapest+2.0)
            previous, current = current, previous
        count = 0
        for value in visits:
            count += value if value >= 0 else 1
        coverage = .3 * count
        return min(1.0, (previous[-1]+coverage)/(len(ref)+coverage))
    return recurrence


_compiled = None


def eed(hypothesis, reference):
    global _compiled
    import numpy as np
    if _compiled is None:
        _compiled = compile_eed()
    return _compiled(np.array([ord(c) for c in hypothesis], dtype=np.int32),
                     np.array([ord(c) for c in reference], dtype=np.int32))
