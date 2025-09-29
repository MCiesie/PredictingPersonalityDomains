import torch
import numpy as np

from Repo.PredictingPersonalityDomains.data.dataset import NUM_FACETS
from Repo.PredictingPersonalityDomains.training.metrics import compute_regression_metrics, compute_classification_metrics


def calculate_baseline_mean(train_dataset, test_dataset):
    y_train = train_dataset.labels.numpy()
    y_test = test_dataset.labels.numpy()

    # Mean PID-5 scores as baseline
    train_means = np.nanmean(y_train, axis=0)

    y_pred_baseline = np.tile(train_means, (len(y_test), 1))

    metrics = compute_regression_metrics(torch.from_numpy(y_pred_baseline), torch.from_numpy(y_test),
                                         train_dataset.means, train_dataset.stds, baseline=True)
    print("Baseline MAE:", metrics["mae"])

    print("Per-domain baseline MAE:", metrics["per_domain_mae"])


def calculate_baseline_majority(train_dataset, test_dataset, num_facets=NUM_FACETS):
    y_train = train_dataset.labels.numpy()  # shape (N_train, num_facets)
    y_test = test_dataset.labels.numpy()  # shape (N_test, num_facets)

    # Majority class per facet (ignoring NaNs)
    majority_classes = []
    for i in range(num_facets):
        facet_labels = y_train[:, i]
        facet_labels = facet_labels[~np.isnan(facet_labels)]
        if len(facet_labels) == 0:
            majority_classes.append(-1)  # no labels available
        else:
            counts = np.bincount(facet_labels.astype(int))
            majority_classes.append(np.argmax(counts))
    majority_classes = np.array(majority_classes)

    # Predict majority class for every sample in test set
    y_pred_baseline = np.tile(majority_classes, (len(y_test), 1))

    # Calculate metrics
    metrics = compute_classification_metrics(torch.from_numpy(y_pred_baseline), torch.from_numpy(y_test), baseline=True)

    print("Baseline Accuracy:", metrics["accuracy"])
    print("Baseline class MAE:", metrics["mae"])
    print("Baseline per-facet Accuracy:", metrics["per_facet_accuracy"])
    print("Baseline per-facet MAE:", metrics["per_facet_mae"])
    print("Baseline Macro-F1:", metrics["per_facet_f1"])
    print("Baseline QWK:", metrics["per_facet_qwk"])
