"""
Runtime noise suppression helpers.
"""

from __future__ import annotations

import os
import warnings


def suppress_runtime_warnings() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

    warnings.filterwarnings(
        "ignore",
        message=r"SymbolDatabase\.GetPrototype\(\) is deprecated\..*",
        category=UserWarning,
        module=r"google\.protobuf\.symbol_database",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"X does not have valid feature names, but RandomForestClassifier was fitted with feature names",
        category=UserWarning,
        module=r"sklearn\.utils\.validation",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*",
        category=UserWarning,
        module=r"tensorflow|keras",
    )
