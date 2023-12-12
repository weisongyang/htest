import jax.numpy as np
from scipy.stats import norm
from jax import jit, vmap


def get_jit_kernel_helper(bandwidth):

    @jit
    def exp_quadratic(x1, x2):
        return np.exp(- (1/bandwidth) * np.sum((x1 - x2)**2))

    @jit
    def cov_map(xs, xs2=None):
        if xs2 is None:
            return vmap(lambda x: vmap(lambda y: exp_quadratic(x, y))(xs))(xs)
        else:
            return vmap(lambda x: vmap(lambda y: exp_quadratic(x, y))(xs))(xs2).T

    def kernel_helper(X, x0):
        return cov_map(X, x0)

    return kernel_helper

@jit
def eucl(x1, x2):
    return np.sum((x1 - x2)**2, axis=1)

@jit
def jaxcdist(X, x0):
    return vmap(lambda y: eucl(X, y))(x0)

@jit
def loglik(y, p):
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    return (y * np.log(p) + (1-y) * np.log(1-p))

