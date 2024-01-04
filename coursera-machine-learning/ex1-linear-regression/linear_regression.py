from itertools import chain

import pandas as pd

class LinearRegression:

    def __init__(self, X, Y, initial_theta=None, alpha=0.01, iterations=1500) -> None:
        self.mus = X.mean()
        self.stds = X.std()
        self.X = self._normalize_features(X)
        self.Y = Y
        self.theta = [initial_theta]
        if initial_theta is None:
            self.theta = [{**{col: 0 for col in X.columns}, **{'ones': 0}}]
        self.alpha = alpha
        self.iterations = iterations
        self.train()    
    
    def _normalize_features(self, x):
        return (x - self.mus) / self.stds

    def hypothesis(self, theta):
        return theta['ones'] + sum([self.X[k] * v for k, v in theta.items() if k != 'ones'])

    def cost(self, theta):
        return (1 / (2 * len(self.X))) * (self.hypothesis(theta) - self.Y).pow(2).sum()

    def gradient_descent(self, theta):
        return {
            **{'ones': theta['ones'] - (self.alpha / len(self.X)) * (self.hypothesis(theta) - self.Y).sum()},
            **{
                k: v - (self.alpha / len(self.X)) * ((self.hypothesis(theta) - self.Y) *(self.X[k])).sum()
                for k, v in theta.items()
                if k != 'ones'
            }
        }

    def train(self):
        for i in range(self.iterations):
            self.theta.append(self.gradient_descent(self.theta[-1]))

    def predict(self):
        return self.X.assign(ones=1).apply(lambda x: x.dot(pd.Series(self.theta[-1])), axis=1)
