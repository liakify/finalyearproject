import numpy as np
import pandas as pd


# File IO

import os
import glob

def load_dataset(folder, id):
    X = np.load(f'{folder}/{id}data.npy')
    y = np.load(f'{folder}/{id}label.npy').astype(dtype=np.int32)
    return X, y

def get_ids(folder):
    ans = []
    for s in glob.glob(f"{folder}/*data.npy"):
        id = os.path.basename(s)
        id = os.path.splitext(id)[0]
        ans.append(id[:-4])
    return ans


# Model evaluation function

from sklearn.model_selection import StratifiedKFold, LeaveOneOut

def evaluate(X, y, model, random_state=None, return_errors=False):
    n_splits = 10
    counts = np.unique(y, return_counts=True)[1]
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) if np.any(counts >= n_splits) else LeaveOneOut()
    errors = []
    for train, test in folds.split(X, y):
        model.fit(X[train], y[train])
        y_pred = model.predict(X[test])
        errors.append(test[y_pred != y[test]])

    errors = np.concatenate(errors)
    accuracy = 1 - (len(errors) / len(y))
    if return_errors:
        return accuracy, errors
    else:
        return accuracy

# Expertise space calculator

def compute_expertise_space(models, folder, ids=None):
    if ids is None:
        ids = get_ids(folder)
    acc = []
    for id in ids:
        X, y = load_dataset(folder, id)
        acc.append([evaluate(X, y, model) for model in models.values()])
    return pd.DataFrame(acc, columns=models.keys(), index=ids)



# Uniformity test

from scipy.stats import chisquare

def uniformity_test(X):
    n, p = X.shape
    n_intervals = 2
    total_bins = n_intervals ** p
    Xt = np.digitize(X, bins=np.arange(0, 1, 1 / n_intervals))
    counts = np.unique(Xt, return_counts=True, axis=0)[1]
    total_counts = np.zeros(total_bins, dtype=np.int64)
    total_counts[:counts.size] = counts
    return chisquare(total_counts).pvalue


# Meta feature extractor

from sklearn.neighbors import NearestNeighbors
# 10-bin beta values by Chen et al 2020
def extract_metafeatures(X, y):
    n, p = X.shape
    nn = NearestNeighbors()
    nn.fit(X)
    labels, label_counts = np.unique(y, return_counts=True)
    nn_dist, nn_idx = nn.kneighbors(X, n_neighbors=n, return_distance=True)
    nn_dist = nn_dist[:, 1:]
    nn_idx = nn_idx[:, 1:]
    inv_dist = 1 / (1 + nn_dist)
    nn_label = y[nn_idx]
    
    criteria = np.equal(nn_label, y[:, np.newaxis])
    for i in range(labels.size):
        idx = np.where(y == labels[i])
        criteria[idx, label_counts[i] - 1:] = False
    
    beta = (inv_dist * criteria).sum(axis=1) / inv_dist.sum(axis=1)
    return np.histogram(beta, bins=10, range=(0, 1), density=True)[0]


# Meta label generator

from scipy.stats import ttest_ind

def get_metalabel(X, y, models, ttest_samples=10):
    acc = []
    for i in range(ttest_samples):
        acc.append([evaluate(X, y, model) for model in models.values()])
    acc = np.stack(acc)
    pvalues = ttest_ind(acc[:, acc.mean(axis=0).argmax()], acc, alternative='greater').pvalue
    return pvalues > 0.05 


# Actions

# Resamples with replacement but with double the probability for error classifications
def resample(X, y, errors):
    n_samples = len(y)
    weights = np.array([1 / n_samples] * n_samples)
    weights[errors] *= 2
    weights /= np.sum(weights)
    selection = np.random.choice(n_samples, size=n_samples, p=weights)
    # selection = np.unique(selection)
    return X[selection], y[selection]


def invert_errors(X, y, model):
    acc, errors = evaluate(X, y, model)
    n_errors = len(errors)
    selection = np.random.choice(errors, size=n_errors // 10)
    y[selection] = 1 - y[selection]
    return X, y

def invert_random(X, y):
    n_samples = len(y)
    selection = np.random.choice(n_samples, size=n_samples // 10)
    y[selection] = 1 - y[selection]
    return X, y


# Plotting functions

import matplotlib.pyplot as plt

def plot(X, y, title=None):
    fig, ax = plt.subplots()
    colours = ('red', 'blue')
    for label in range(2):
        ax.scatter(x=X[y==label, 0], 
                y=X[y==label, 1], 
                c=colours[label], 
                s=20,
                label=label)

    ax.set(xlabel='X',
        ylabel='y',
        title=title)
    ax.legend(loc='upper right')

    plt.show()