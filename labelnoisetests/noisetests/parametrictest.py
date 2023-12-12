import numpy as np
from sklearn.linear_model import LogisticRegression

from scipy.stats import norm

def get_pvalue(x, mean, scale):
    z = (x - mean)/scale
    cdf = norm(0, 1).cdf(z)[0]
    if z > 0:
        pvalue = 1 - cdf
    else:
        pvalue = cdf
    return 2*pvalue

def parametric_test(
    X: np.ndarray,        # (n_samples, n_features)
    y: np.ndarray,        # (n_samples, )
    anchors: np.ndarray   # (n_anchors, n_features) 
    ):
    n_samples, n_features = X.shape
    n_anchors = anchors.shape[0]
    
    nx = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
    nanchors = np.concatenate((np.ones((n_anchors, 1)), anchors), axis=1)
    
    clf = LogisticRegression(
        fit_intercept=False,
        C=1e5,
        tol=1e-6,
        max_iter=int(1e5)
    ).fit(nx, y)
    
    xpreds = clf.predict_proba(nx)
    anchorpreds = clf.predict_proba(nanchors)[:, 1]
    
    D = np.diag(xpreds[:, 0]*xpreds[:, 1])
    
    # inverse of variance of \beta
    # https://imai.fas.harvard.edu/teaching/files/mle.pdf -- p17
    V = nx.T @ D @ nx

    # variance of \beta
    Q=np.linalg.inv(V)
    
    # All at once: see eq. 15
    mean_anchor = nanchors.mean(axis=0).reshape(-1, 1)
    mean_anchor_pred = anchorpreds.mean(axis=0)

    omega = mean_anchor.T @ Q @ mean_anchor
    scale = np.sqrt(omega)[0] / 4
    pvalue = get_pvalue(mean_anchor_pred, 0.50, scale)
    
    return pvalue

def nonparametric_test(
    X: np.ndarray,        # (n_samples, n_features)
    y: np.ndarray,        # (n_samples, )
    anchors: np.ndarray   # (n_anchors, n_features) 
    ):
    n_samples, n_features = X.shape
    n_anchors = anchors.shape[0]
    
    nx = np.concatenate((np.ones((n_samples, 1)), X), axis=1)
    nanchors = np.concatenate((np.ones((n_anchors, 1)), anchors), axis=1)
    
    clf = LogisticRegression(
        fit_intercept=False,
        C=1e5,
        tol=1e-6,
        max_iter=int(1e5)
    ).fit(nx, y)
    
    xpreds = clf.predict_proba(nx)
    anchorpreds = clf.predict_proba(nanchors)[:, 1]
    
    D = np.diag(xpreds[:, 0]*xpreds[:, 1])
    
    # inverse of variance of \beta
    # https://imai.fas.harvard.edu/teaching/files/mle.pdf -- p17
    V = nx.T @ D @ nx

    # variance of \beta
    Q=np.linalg.inv(V)
    
    # All at once: see eq. 15
    mean_anchor = nanchors.mean(axis=0).reshape(-1, 1)
    mean_anchor_pred = anchorpreds.mean(axis=0)

    omega = mean_anchor.T @ Q @ mean_anchor
    scale = np.sqrt(omega)[0] / 4
    pvalue = get_pvalue(mean_anchor_pred, 0.50, scale)
    
    return pvalue