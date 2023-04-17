
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

class LinearRegression:

    def __init__(self, X, Y, initial_theta=(0, 0), alpha=0.01, iterations=1500) -> None:
        self.X = X
        self.Y = Y
        self.theta = [initial_theta]
        self.alpha = alpha
        self.iterations = iterations
        self.train()

    def hypothesis(self, theta):
        return theta[0] + self.X * theta[1]

    def cost(self, theta):
        return (1 / (2 * len(self.X))) * (self.hypothesis(theta) - self.Y).pow(2).sum()

    def gradient_descent(self, theta):
        return (
            theta[0] - (self.alpha / len(self.X)) * (self.hypothesis(theta) - self.Y).sum(),
            theta[1] - (self.alpha / len(self.X)) * ((self.hypothesis(theta) - self.Y) * self.X).sum(),
        )

    def train(self):
        for i in range(self.iterations):        
            self.theta.append(self.gradient_descent(self.theta[-1]))

    def predict(self, x):
        return pd.Series((1, x)).dot(pd.Series(self.theta[-1]))

# Read data into pandas.DataFrame
data = pd.read_csv(Path(__file__).parent / 'food_truck_data.csv')

# Initialize linear regression
lr = LinearRegression(X=data['pop'], Y=data['profit'])

# Make predictions
data['prediction'] = data['pop'].apply(lr.predict)

# Visalize Predictions
fig, ax = plt.subplots()
ax.scatter(x=data['pop'], y=data['profit'], marker='x', color='red')
ax.set_ylabel('Profit ($1000s)')
ax.set_xlabel('City Population (1000s)')
ax.set_title('Food Truck Profits')
ax.plot(data['pop'], data['prediction'])
fig.savefig(Path(__file__).parent / 'prediction.png')


# Visalize Cost Curve
costs_fig, costs_ax = plt.subplots()
costs_ax.plot([lr.cost(theta) for theta in lr.theta])
costs_ax.set_ylabel('Cost (Mean Sum of Squared Error Terms)')
costs_ax.set_xlabel('Iteration')
costs_ax.set_title('Costs Per Iteration')
costs_fig.savefig(Path(__file__).parent / 'costs_per_iteration.png')

# Visalize Gradient Descent
grad_fig, grad_ax = plt.subplots(subplot_kw={'projection': '3d'})
grad = pd.DataFrame({
    'theta_0': np.random.uniform(-10, 10, size=1000), 
    'theta_1': np.random.uniform(4, -4, size=1000),
})
grad['cost'] = grad.apply(lambda x: lr.cost([x['theta_0'], x['theta_1']]), axis=1)
grad_ax.plot_trisurf(grad['theta_1'], grad['theta_0'], grad['cost'], cmap=cm.jet, linewidth=0)
grad_ax.set_ylabel('theta_0')
grad_ax.set_xlabel('theta_1')
grad_ax.set_zlabel('Cost')
grad_fig.savefig(Path(__file__).parent / 'gradient_descent.png')
