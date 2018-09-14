#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:02:21 2018

@author: oster
"""

import numpy as np
import seaborn as sns
import pandas as pd

def p_x_given_y(y, mus, sigmas):
    mu = mus[0] + sigmas[1, 0] / sigmas[0, 0] * (y - mus[1])
    sigma = sigmas[0, 0] - sigmas[1, 0] / sigmas[1, 1] * sigmas[1, 0]
    return np.random.normal(mu, sigma)


def p_y_given_x(x, mus, sigmas):
    mu = mus[1] + sigmas[0, 1] / sigmas[1, 1] * (x - mus[0])
    sigma = sigmas[1, 1] - sigmas[0, 1] / sigmas[0, 0] * sigmas[0, 1]
    return np.random.normal(mu, sigma)


def gibbs_sampling(mus, sigmas, iter=100000):
    samples = np.zeros((iter, 2))
    y = np.random.rand() * 10

    for i in range(iter):
        x = p_x_given_y(y, mus, sigmas)
        y = p_y_given_x(x, mus, sigmas)
        samples[i, :] = [x, y]

    return samples


#if __name__ == '__main__':
mus = np.array([5, 5])
sigmas = np.array([[1, .9], [.9, 1]])
samples = gibbs_sampling(mus, sigmas)
trial  = pd.DataFrame(np.random.randn(2000,2), columns =["x","y"])
#sns.jointplot(x="x",y="y", data=trial)
sns.jointplot(samples[:, 0], samples[:, 1])
print("done")