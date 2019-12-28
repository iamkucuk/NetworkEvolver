import numpy as np


def evaluate_probas(y_true: np.ndarray, y_proba: np.ndarray):
    """
    Evaluation metric (Precision-Recall Curve) for probability evaluation
    :param y_true: Ground truth
    :param y_proba: Output of model. If the model output is not between 0 and 1, a sigmoid function will be applied.
    :return: Precision and recall lists
    """

    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    if (np.min(y_proba) < 0) or (np.max(y_proba) > 1):
        y_proba = sigmoid(y_proba)

    thresholds = np.arange(0.0, 1.0, .01)

    # pos = sum(y_true)
    # neg = len(y_true) - pos

    recalls = []
    precisions = []

    for threshold in thresholds:
        confusion_matrix = evaluate_predictions(y_true=y_true, y_pred=y_proba, threshold=threshold)
        tn, fn = confusion_matrix[0]
        fp, tp = confusion_matrix[1]
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (tp + fn))

    return precisions, recalls


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = .00001) -> np.ndarray:
    """
    Outputs confusion matrix.
    :param threshold: Default .000001 - Threshold be applied if y_pred is not predicted as a class (0 or 1)
    :param y_true: Ground truth
    :param y_pred: Output of model. Can be both prediction or predictid probabiltiy.
    :return: 2x2 Confusion matrix
    """
    y_pred = np.copy(y_pred)
    y_pred[y_pred > threshold] = 1
    y_pred[y_pred < threshold] = 0
    tp = np.sum(np.logical_and(y_pred == 1, y_true == 1))
    fp = np.sum(np.logical_and(y_pred == 1, y_true == 0))
    tn = np.sum(np.logical_and(y_pred == 0, y_true == 0))
    fn = np.sum(np.logical_and(y_pred == 0, y_true == 1))
    return np.array([[tn, fn], [fp, tp]])
