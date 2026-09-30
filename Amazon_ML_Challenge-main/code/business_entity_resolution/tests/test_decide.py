import itertools

import numpy as np
import polars as pl
import pytest

from er.decide import DecisionParams, expected_f_exact, select, select_exact


def _brute(p, k, beta2=0.25):
    """E[F_beta] of predicting the first k items, by enumerating every truth assignment."""
    n, e = len(p), 0.0
    for truth in itertools.product([0, 1], repeat=n):
        pr = np.prod([pi if t else 1 - pi for pi, t in zip(p, truth)])
        tp, nt = sum(truth[:k]), sum(truth)
        if k == 0:
            f = 1.0 if nt == 0 else 0.0
        elif nt == 0 or tp == 0:
            f = 0.0
        else:
            prec, rec = tp / k, tp / nt
            f = (1 + beta2) * prec * rec / (beta2 * prec + rec)
        e += pr * f
    return e


@pytest.mark.parametrize("p", [[0.6], [0.9, 0.5], [0.95, 0.7, 0.3, 0.05], [0.5, 0.5, 0.5, 0.5, 0.5]])
def test_expected_f_exact_matches_bruteforce(p):
    ef = expected_f_exact(np.array([p]))
    for k in range(len(p) + 1):
        assert ef[0, k] == pytest.approx(_brute(p, k), abs=1e-9)


def _df(rows):
    return pl.DataFrame(rows, schema={"s1_idx": pl.UInt32, "src": pl.UInt8, "t_idx": pl.UInt32, "p": pl.Float64},
                        orient="row")


def test_select_exact_decisions():
    d = _df([(0, 2, 10, 0.95), (0, 3, 11, 0.9), (0, 2, 12, 0.08),   # clear pair + noise
             (1, 2, 20, 0.30),                                        # doubtful single -> empty
             (2, 2, 30, 0.97)])                                       # confident single
    sel = select_exact(d, DecisionParams())
    got = set(map(tuple, sel.select("s1_idx", "t_idx").rows()))
    assert got == {(0, 10), (0, 11), (2, 30)}


def test_select_exact_respects_target_exclusivity():
    # target 10 is wanted by S1 0 (0.9) and S1 1 (0.7): only S1 0 may take it
    d = _df([(0, 2, 10, 0.9), (1, 2, 10, 0.7), (1, 3, 11, 0.95)])
    sel = select_exact(d, DecisionParams())
    got = set(map(tuple, sel.select("s1_idx", "t_idx").rows()))
    assert (1, 10) not in got and (0, 10) in got and (1, 11) in got


def test_exact_and_ratio_agree_on_easy_cases():
    d = _df([(0, 2, 1, 0.99), (0, 3, 2, 0.98), (1, 2, 3, 0.01)])
    a = set(map(tuple, select_exact(d, DecisionParams()).rows()))
    b = set(map(tuple, select(d, DecisionParams()).rows()))
    assert a == b == {(0, 2, 1), (0, 3, 2)}
