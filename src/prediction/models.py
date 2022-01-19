import numpy as np
from sklearn.linear_model import Lasso, Ridge


class LassoWrapper():
    def __init__(self,
                 search_type=None,
                 model_params={'fit_intercept': True}):
        self.model_name = "lasso"
        if search_type is None:
            self.search_type = 'random'
        else:
            self.search_type = search_type
        self.param_grid = {'alpha': np.linspace(0.001, 50, 100)}
        if model_params is None:
            self.ModelClass = Lasso()
        else:
            self.ModelClass = Lasso(**model_params)


class RidgeWrapper():
    def __init__(self,
                 search_type=None,
                 model_params={'fit_intercept': True}):
        self.model_name = "ridge"
        if search_type is None:
            self.search_type = 'random'
        else:
            self.search_type = search_type
        self.param_grid = {'alpha': np.linspace(0.001, 50, 100)}
        if model_params is None:
            self.ModelClass = Ridge()
        else:
            self.ModelClass = Ridge(**model_params)

class peLassoWrapper():
    def __init__(self):
        self.model_name = "pelasso"
        self.search_type = 'grid'
        self.Lasso = LassoWrapper(search_type=self.search_type)
        self.Ridge = RidgeWrapper(search_type=self.search_type)