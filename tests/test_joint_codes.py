"""Integer mixed-radix joint codes for combination units (lever #2, KI-019/KI-018 foundation).

Locks the contract the gather/fit/OOF paths rely on:
- a combination maps to one stable int64 code; distinct combinations get distinct codes;
- an unknown component forces the whole row to the -1 sentinel (-> downstream unknown fallback);
- decode inverts the code back to the component-value tuple (used to keep categories_ as tuples);
- when the radix product overflows int64 the plan declines the int path and the encoder falls back
  to the object-tuple key build, producing identical output.
"""

import numpy as np
import pandas as pd

from catstat import TargetEncoder
from catstat._cross_fit import build_joint_keyplan, decode_joint, joint_codes
from catstat._validation import normalize_keys


def _norm(values):
    return normalize_keys(np.asarray(values, dtype=object))[0]


def test_joint_codes_stable_distinct_and_decodable():
    a = _norm(["x", "y", "x", "y", "z"])
    b = _norm(["p", "p", "q", "q", "p"])
    plan = build_joint_keyplan([a, b])
    assert plan.use_int
    codes = joint_codes(plan, [a, b])
    assert codes.dtype == np.int64

    combos = list(zip(a, b))
    # same combination <=> same code (a bijection over observed combos)
    for i in range(len(combos)):
        for j in range(len(combos)):
            assert bool(codes[i] == codes[j]) == (combos[i] == combos[j])
    # decode inverts the mixed-radix code back to the component-value tuples
    assert decode_joint(plan, codes) == combos


def test_joint_codes_unknown_component_is_minus_one_sentinel():
    a = _norm(["x", "y"])
    b = _norm(["p", "q"])
    plan = build_joint_keyplan([a, b])
    # 'z' never appeared in component a at fit -> the whole (z, p) row collapses to the -1 sentinel
    codes = joint_codes(plan, [_norm(["x", "z"]), _norm(["p", "p"])])
    assert codes[0] >= 0
    assert codes[1] == -1


def test_build_joint_keyplan_overflow_declines_int_path(monkeypatch):
    monkeypatch.setattr("catstat._cross_fit._INT64_MAX", 3)  # radix product 2*2=4 > 3
    plan = build_joint_keyplan([_norm(["x", "y"]), _norm(["p", "q"])])
    assert plan.use_int is False


def _combo_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.choice(list("xyz"), n), "b": rng.choice(list("pqr"), n)})
    return X, rng.normal(size=n)


def test_overflow_fallback_matches_int_path_and_keeps_tuple_categories(monkeypatch):
    X, y = _combo_data()
    kw = dict(
        cols=["a", "b"],
        multi_feature_mode="combination",
        smooth=0.0,
        cv=5,
        random_state=0,
        output="numpy",
    )
    enc_int = TargetEncoder(**kw).fit(X, y)
    assert enc_int._unit_keyplans  # int joint-code path active
    int_oof = TargetEncoder(**kw).fit_transform(X, y)

    monkeypatch.setattr("catstat._cross_fit._INT64_MAX", 3)  # force the tuple fallback
    enc_tup = TargetEncoder(**kw).fit(X, y)
    assert not enc_tup._unit_keyplans  # no plan stored -> _unit_keys builds tuples
    tup_oof = TargetEncoder(**kw).fit_transform(X, y)

    # the int path and the tuple fallback are the same encoder, only a different key representation
    assert np.allclose(int_oof, tup_oof)
    assert np.array_equal(enc_int.categories_["a+b"], enc_tup.categories_["a+b"])
    # categories_ holds decoded category VALUES (object), never raw integer codes
    assert enc_int.categories_["a+b"].dtype == object
    assert isinstance(enc_int.categories_["a+b"].ravel()[0], str)
