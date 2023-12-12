import numpy as onp
import jax.numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample
from sklearn.base import BaseEstimator, ClassifierMixin

from copy import deepcopy

from typing import Optional, Callable, Union, List
from jax import random

from sklearn.model_selection import LeaveOneOut, GridSearchCV
import itertools as it

from locreg.supp import correct_predictions, loglik, get_jit_kernel_helper, jaxcdist


def gridsearch_fn(X, y, param_grid, n_instances=-1):
    gs_param_grid = {
        'kernel_kwargs': [
            dict(zip(param_grid.keys(), row))
            for row in it.product(*param_grid.values())
        ]
    }

    indices_to_use = range(X.shape[0])
    if n_instances > 0:
        indices_to_use = onp.random.choice(indices_to_use, n_instances, replace=False)
      
    loo = LeaveOneOut()
    loo.get_n_splits(onp.array(X)[indices_to_use, :])

    gridsearch = GridSearchCV(LocalRegression(), gs_param_grid, cv=loo)
    gridsearch.fit(X, y)
    
    return gridsearch


class BaseLocalRegressor:
    def __init__(self,
        X: np.ndarray,                      # (n_samples, n_features)
        y: np.ndarray,                      # (n_samples,)
        transform: PolynomialFeatures,      # feature transformation
        kernel_func: str,                   # which kernel function to use
        base_model_type: str,               # Options: logreg
        kernel_kwargs: dict,                # kernel parameters
        ):

        self.base_model_type = base_model_type
        if self.base_model_type == 'logreg':
            self.model = LogisticRegression(solver='lbfgs', max_iter=10000, C=10e10, fit_intercept=False)
        else:
            raise ValueError(f'base_model_type: {self.base_model_type} not recognised.')

        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        
        self.transform = transform
        self.kernel_func = kernel_func
        self.kernel_kwargs = kernel_kwargs

        if self.kernel_func == 'rbf':
            self._kernel_func = self.__rbf_kernel
            self._kernel_const = (1/((np.sqrt(2*np.pi))))
            self._rbf_kernel_helper = get_jit_kernel_helper(self.kernel_kwargs['bandwidth'])
        elif self.kernel_func == 'tricube':
            self._kernel_func = self.__tri_cube_kernel
            self._kernel_const = 35/32
        
        self._predict_fn = self._make_predict_fn()

    @classmethod
    def _make_new(cls, kwargs):
        return cls(**kwargs)

    def _make_predict_fn(self) -> Callable:
        def predict(x):
            if self.base_model_type == 'logreg':
                pr = self.model.predict_proba
            return pr(self.transform(x))
        return predict

    def fit(
        self,
        x0: np.ndarray,  # (1, n_features)
        ):

        self.x0 = x0.reshape(1, -1)
        self.weights = self._kernel_func(self.x0)

        self.model.fit(
            self.transform(self.X - self.x0),
            self.y,
            sample_weight=self.weights,
            )

        self.pred0 = self._predict_fn(np.zeros_like(self.x0))

        return self

    def __rbf_kernel(
        self,
        x0: np.ndarray,
        ) -> np.ndarray:

        const = 1/((np.sqrt(2*np.pi)))
        weights = self._rbf_kernel_helper(self.X, x0)
        weights *= const
        return np.squeeze(weights)
    
    def __tri_cube_kernel(
        self,
        x0: np.ndarray
        ) -> np.ndarray:

        points_cnt = self.kernel_kwargs['kernel_size']
        if points_cnt <= 1:
            points_cnt = int(self.X.shape[0] * points_cnt)
        distances = onp.array(jaxcdist(self.X, x0)).ravel()
        kernel_points_indices = onp.argpartition(
            distances, points_cnt, axis=0)[:points_cnt]
        max_distance = np.max(distances[kernel_points_indices])
        relative_distance = distances/max_distance
        result = (relative_distance <= 1)*(1-relative_distance**3)**3
        return np.squeeze(result)


class LocalRegression(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        kernel_kwargs: Optional[dict] = None,
        kernel_func: str = 'rbf',
        base_model_type: str = 'logreg',
        transform: Optional[Callable] = None,
        ) -> None:

        self.transform = transform if transform is not None else PolynomialFeatures(degree=1)

        self.kernel_func = kernel_func
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {'bandwidth': 0.10}

        self.base_model_type = base_model_type

        self._state_dict_objects = [
            'kernel_kwargs',
            'transform',
            'kernel_func',
            'base_model_type',
            'X'
        ]

        self._kwargs_keys = [
            'X',
            'y',
            'transform',
            'kernel_func',
            'base_model_type',
            'kernel_kwargs',
        ]

    def __str__(self) -> str:
        s = 'LocalRegression('
        for item in self._state_dict_objects:
            s += f'{item}: {getattr(self, item)}, '
        s+=')'
        return s

    def __repr__(self) -> str:
        return self.__str__()

    def state_dict(self) -> dict:
        return self.kwargs
        
    def load_state_dict(
        self,
        state_dict: dict,
        ):

        self.kwargs = {
            key: state_dict[key]
            for key in self._kwargs_keys
        }

        for item in self._state_dict_objects:
            setattr(self, item, state_dict[item])

        return self

    def score(
        self,
        X: np.ndarray,          # (n_samples, n_features)
        y: np.ndarray = None,   # (n_samples,)
        ) -> float:

        return loglik(y, self.predict(X)[:, 1])[0]

    def predict(
        self,
        X: np.ndarray,                          # (n_samples, n_features)
        return_models: Optional[bool] = False,  # reeturn models, instead of just predictions
        ) -> Union[List[Callable], np.ndarray]:

        models = [
            BaseLocalRegressor._make_new(self.kwargs).fit(X[i:i+1, :])
            for i in range(X.shape[0])
        ]
        if return_models:
            return models
        return np.vstack([
            model.pred0
            for model in models
            ])
    
    def fit(
        self,
        X: np.ndarray,          # (n_samples, n_features)
        y: np.ndarray = None,   # (n_samples,)
        ):
        
        self.transform.fit(X)

        self.X = X
        self.kwargs = {
            'X': X,
            'y': y,
            'transform': self.transform.transform,
            'kernel_func': self.kernel_func,
            'kernel_kwargs': self.kernel_kwargs,
            'base_model_type': self.base_model_type,
        }

        return self


class BootstrapLocalRegression:
    def __init__(
        self,
        kernel_kwargs_grid: Optional[dict] = None,
        kernel_func: Optional[str] = 'rbf',
        n_instances: Optional[int] = -1,
        n_estimators: Optional[int] = 1,
        bootstrap_type: Optional[str] = 'nonparametric',
        hparam_sweep_once: Optional[bool] = True,
        base_model_type: Optional[str] = 'logreg',
        ):

        __default_param_grids = {
            'rbf': {
                'bandwidth': [0.10],
            },
            'tricube': {
                'kernel_size': [0.10],
            }
        }

        self.kernel_func = kernel_func
        self._param_grid = __default_param_grids[self.kernel_func]
        if kernel_kwargs_grid is not None:
            for key, value in kernel_kwargs_grid.items():
                self._param_grid[key] = value if isinstance(value, list) else [value]

        self.n_instances = n_instances
        self.n_estimators = n_estimators
        self.bootstrap_type = bootstrap_type
        self.hparam_sweep_once = hparam_sweep_once
        self.base_model_type = base_model_type

        self.transform = PolynomialFeatures(degree=1)

        self._state_dict_objects = [
            '_param_grid',
            'n_instances',
            'n_estimators',
            'bootstrap_type',
            'hparam_sweep_once',
            'base_model_type',
        ]

    def __str__(self):
        s = 'BootstrapLocalRegression('
        for item in self._state_dict_objects:
            s += f'{item}: {getattr(self, item)}, '
        s+=')'
        return s

    def __repr__(self):
        return self.__str__()

    def state_dict(self):
        return {
            'state_dicts': [model.state_dict() for model in self.models],
            **{
                item: getattr(self, item)
                for item in self._state_dict_objects
            }
        }
        
    def load_state_dict(
        self,
        state_dict: dict,
        ):

        for item in self._state_dict_objects:
            if item == 'state_dicts': continue
            setattr(self, item, state_dict[item])

        self.models = [
            LocalRegression().load_state_dict(state_dict)
            for state_dict in state_dict['state_dicts']
        ]
        return self

    def _parametric_bootstrap(
        self,
        X: np.ndarray,          # (n_samples, n_features)
        ):
        base_model_predictions = self.models[0].predict(X)
        seed = onp.random.randint(20000)
        key = random.PRNGKey(seed)
        newy = random.bernoulli(key, base_model_predictions)
        return X, newy
    
    def _resample(self):

        if self.bootstrap_type == 'nonparametric':
            return resample(deepcopy(self.X), deepcopy(self.y))
        elif self.bootstrap_type == 'parametric':
            return self._parametric_bootstrap(deepcopy(self.X))
        
    def fit(
        self,
        X: np.ndarray,          # (n_samples, n_features)
        y: np.ndarray = None,   # (n_samples,)
        ):

        self.X = X
        self.y = y

        gridsearch = gridsearch_fn(self.X, self.y, param_grid=self._param_grid, n_instances=self.n_instances)
        self.models = [gridsearch.best_estimator_]
            
        for _ in range(self.n_estimators):
            nx, ny = self._resample()

            if self.hparam_sweep_once:
                model = LocalRegression(
                    **gridsearch.best_params_,
                    transform=self.transform,
                    ).fit(nx, ny)
            else:
                gridsearch = gridsearch_fn(nx, ny, param_grid=self._param_grid, n_instances=self.n_instances)
                model = gridsearch.best_estimator_

            self.models.append(model)
        return self

    def predict(
        self,
        X: np.ndarray,                      # (n_samples, n_features)
        to_return: Optional[bool] = 'all',
        ):

        predictions = np.vstack([
            model.predict(X)[:, 1]
            for model in self.models
        ])

        if to_return in ['all', 'a']:
            return predictions
        elif to_return in ['mean', 'm']:
            return np.mean(predictions, axis=1)
        elif to_return in ['corrected', 'c']:
            return correct_predictions(predictions)
        elif to_return in ['correctred_and_std', 's']:
            return correct_predictions(predictions, True)
        else:
            raise ValueError(f'to_return: {to_return} not recognised.')
