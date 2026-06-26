"""Documented scikit-learn ``check_estimator`` subset (KI-012).

catstat's encoders are *categorical* transformers (string/categorical input, the supervised one
needs ``y``), so a chunk of the generic estimator suite doesn't apply. We run the full suite and
**waive** the inapplicable checks via ``expected_failed_checks``, each with a one-line reason — so
the applicable checks are enforced and the gaps are explicit rather than silent.

Requires scikit-learn >= 1.6 (the ``expected_failed_checks`` API); skipped on the dev box's 1.2.
The estimators use ``cols=[0]`` so the suite's numeric fixtures are encoded (``cols="auto"`` would
reject numeric-only input — itself a categorical-encoder property).
"""

import pytest
import sklearn
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.fixes import parse_version

from catstat import CountEncoder, FrequencyEncoder, TargetEncoder

SKLEARN_GE_16 = parse_version(sklearn.__version__) >= parse_version("1.6")
pytestmark = pytest.mark.skipif(
    not SKLEARN_GE_16, reason="check_estimator expected_failed_checks API needs scikit-learn>=1.6"
)

# Checks waived for all catstat encoders, with the reason each does not apply.
_COMMON_XFAIL = {
    "check_estimator_sparse_tag": "categorical encoder; sparse input unsupported",
    "check_estimator_sparse_array": "categorical encoder; sparse input unsupported",
    "check_estimator_sparse_matrix": "categorical encoder; sparse input unsupported",
    "check_n_features_in_after_fitting": "columns selected by name; n_features not enforced",
    "check_transformer_general": "n_features tolerated in transform (columns by name)",
    "check_fit1d": "1D X is reshaped to one column, not rejected",
    "check_fit2d_predict1d": "1D transform input is reshaped, not rejected",
    "check_estimators_empty_data_messages": "no sklearn-style message on empty input",
}
# Supervised TargetEncoder additionally has y-validation message and 1-sample (folds) gaps.
_TARGET_XFAIL = {
    **_COMMON_XFAIL,
    "check_requires_y_none": "requires-y message differs from sklearn's",
    "check_dtype_object": "unknown-y-label message differs from sklearn's",
    "check_fit2d_1sample": "out-of-fold cross-fitting needs >1 sample",
}
# Unsupervised count/frequency encoders.
_UNSUP_XFAIL = {
    **_COMMON_XFAIL,
    "check_complex_data": "complex input not rejected with a message",
}


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (TargetEncoder(cols=[0]), _TARGET_XFAIL),
        (CountEncoder(cols=[0]), _UNSUP_XFAIL),
        (FrequencyEncoder(cols=[0]), _UNSUP_XFAIL),
    ],
    ids=["TargetEncoder", "CountEncoder", "FrequencyEncoder"],
)
def test_check_estimator_documented_subset(estimator, expected):
    # Raises if any *non-waived* check fails; waived checks (with reasons above) are tolerated.
    check_estimator(estimator, expected_failed_checks=expected)
