import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.model import (
    train_model,
    inference,
    compute_model_metrics,
)


def test_train_model():
    """Test that train_model returns a RandomForestClassifier."""
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
        [6, 5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X, y)

    assert isinstance(model, RandomForestClassifier)


def test_inference():
    """Test that inference returns one prediction per observation."""
    X = np.array([
        [1, 2],
        [2, 1],
        [3, 4],
        [4, 3],
        [5, 6],
        [6, 5],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    model = train_model(X, y)
    preds = inference(model, X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(y)


def test_compute_model_metrics():
    """Test precision, recall, and F1 for perfect predictions."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0

