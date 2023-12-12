import jax.numpy as np
from scipy.stats import norm


def correct_predictions(predictions, return_std=False):
    corrected = 2*predictions[0, :] - predictions[1:, :].mean(axis=0)
    if return_std:
        std = predictions[1:, :].std(axis=0)
        return corrected, std
    return corrected

def logistic(z):
    return 1/(1+np.exp(-z))

def get_var(model, invtype=0):
    Ap = model.transform(model.X - model.x0)
    x_preds = model.clf.predict_proba(Ap)
    eps = 1e-6
    preds = np.clip(x_preds[:, 1], eps, 1 - eps)
    V = np.diag(np.einsum('i,i->i', preds, 1-preds))
    W = np.diag(model.weights)

    vap = V @ Ap
    apw = Ap.T @ W
    J1 = apw @ vap
    if invtype in [0, 2]:  # 22, 3
        J2 = apw @ W @ vap
        if invtype == 2:
            J2 /= 4
    elif invtype == 1:  # 222
        J2 = apw @ W @ np.diag((model.y - preds)**2) @ Ap
    else:
        raise ValueError(f'invtype: {invtype} not recognised.')

    invJ1 = np.linalg.inv(J1)
    var = invJ1 @ J2 @ invJ1

    pr = model.pred0
    pr_var = var[0, 0]
    assert pr_var >= 0

    return pr_var, pr

def normal_theory_interval(preds, alpha):
    m = np.mean(preds, axis=0)
    std = np.std(preds, axis=0)
    
    z_alpha = norm.ppf(1-alpha/2)
    out = {
        'lb': m - z_alpha * std,
        'ub': m + z_alpha * std,
        'm': m,
        'std': std,
    }
    return out

def bootstrap_percentile_interval(preds, alpha):
    lower = 100 * alpha / 2
    upper = 100 * (1 - alpha/2)
    return np.percentile(preds, [lower, 50, upper], axis=0)

def get_preds(models):
    return np.array([
        [
            models[i][j].pred0[0, 1]
            for j in range(len(models[i]))]
        for i in range(len(models))
    ]).T

def get_inv_infmat_m(model, invtype=0, div4=True):
    Ap = model.transform(model.X - model.x0)
    x_preds = model.model.predict_proba(Ap)
    eps = 1e-6
    preds = np.clip(x_preds[:, 1], eps, 1 - eps)
    V = np.diag(np.einsum('i,i->i', preds, 1-preds))
    W = np.diag(model.weights)

    vap = V @ Ap
    apw = Ap.T @ W
    J1 = apw @ vap
    if invtype == 22:
        J2 = apw @ W @ vap
    elif invtype == 222:
        J2 = apw @ W @ np.diag((model.y - preds)**2) @ Ap

    invJ1 = np.linalg.inv(J1)
    var = invJ1 @ J2 @ invJ1

    pr = model.pred0
    if div4:
        pr_var = var[0, 0] #* (pr[0, 0] * pr[0, 1])**2
    else:
        pr_var = var[0, 0] * (pr[0, 0] * pr[0, 1])**2
    
    # print(var, (model.weights**2).sum())
    assert pr_var >= 0   #, print(var, (model.weights**2).sum())

    hJ2 = apw @ np.diag(model.y - preds)

    return invJ1, hJ2, var[0, 0], model.pred0
