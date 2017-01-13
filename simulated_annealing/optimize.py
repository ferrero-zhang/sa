# -*- coding: utf-8 -*-

import sys
import time
from copy import copy
import random
import sklearn.cross_validation as cross_validation
from sklearn.base import clone
import numpy as np
from sklearn.metrics.scorer import get_scorer
from sklearn.externals.joblib import Parallel, delayed
from sklearn.cross_validation import _fit_and_score

class SimulatedAnneal(object):
    def __init__(self, T=0.5, max_iter=10,T_min=0.0001, alpha=0.75,cf=0.8):

        assert alpha <= 1.0
        assert T > T_min
        assert max_iter is not None and max_iter > 0
        self.__T = T
        self.__alpha = alpha
        self.__max_iter = max_iter
        self._cf = cf
        self.__T_min = T_min

    def fit(self):
        # Set up  the initial params
        T = self.__T
        alpha = self.__alpha
        max_iter = self.__max_iter
        cf = self.__cf
        T_min = self.__T_min

        # Computes the acceptance probability as a function of T; maximization
        accept_prob = lambda old, new, T: np.exp((new-old)/T)
        total_iter = 1

        while T > T_min and total_iter < max_iter :
            '''计算 HB......
            '''
            a = accept_prob(old_score, new_score, T)
            if a > cf:
                old_params = new_params
                old_score = new_score
                t_elapsed = dt(time_at_start, time.time())
                iter_ += 1
            '''update data 
            '''
            T *= alpha
        return [] or np.array
        

