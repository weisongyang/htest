import numpy as onp
import jax.numpy as np

from locreg import LocalRegression
from locreg import BootstrapLocalRegression
from locreg.main import gridsearch_fn
from locreg.supp import get_inv_infmat_m

from scipy.stats import norm

def get_pvalue(x, mean, scale):
    z = (x - mean)/scale
    cdf = norm(0, 1).cdf(z)
    if z > 0:
        pvalue = 1 - cdf
    else:
        pvalue = cdf
    return 2*pvalue

def nonparametric_test(clf, anchors, invtype):
    n_anchors = len(anchors)
    _omegas_sum = 0
    _preds_sum = 0
    js = []
    var = 0.
    for anchor in anchors:
        mdl = clf.predict(anchor, return_models=True)

        valid=True
        try:
            invj1, hj2, single_var, pred = get_inv_infmat_m(mdl[0], invtype=invtype)
        except AssertionError as err:
            print(anchor)
            valid=False
            return -1

        js.append([invj1, hj2])

        _preds_sum += pred[0][0]
        assert single_var >= 0, print(single_var)

        var += single_var

    if valid:
        if invtype in [22, 222]:
            for ii in range(n_anchors):
                invj1_0, hj2_0 = js[ii]
                for jj in range(ii+1, n_anchors):
                    invj1_1, hj2_1 = js[jj]
                    cvar = 2 * (invj1_0 @ hj2_0 @ hj2_1.T @ invj1_1.T)[0, 0]
                    # assert cvar > 0, print(cvar)
                    var += cvar

        scale = np.sqrt(var) / 4
        scale = scale / n_anchors
        pvalue = get_pvalue(_preds_sum/n_anchors, 0.50, scale)

    return pvalue