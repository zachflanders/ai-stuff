
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm

from .linear_regression import LinearRegression


# Read data into pandas.DataFrame
data = pd.read_csv(Path(__file__).parent / 'food_truck_data.csv')

# Initialize linear regression
lr = LinearRegression(X=data[['pop']], Y=data['profit'])

# Make predictions
data['prediction'] = lr.predict()


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

# # Visalize Gradient Descent
# grad_fig, grad_ax = plt.subplots(subplot_kw={'projection': '3d'})
# grad = pd.DataFrame({
#     'ones': np.random.uniform(-10, 10, size=1000), 
#     'pop': np.random.uniform(4, -4, size=1000),
# })
# grad['cost'] = grad.apply(lambda x: lr.cost{ones: x['ones'], x['pop']]), axis=1)
# grad_ax.plot_trisurf(grad['ones'], grad['pop'], grad['cost'], cmap=cm.jet, linewidth=0)
# grad_ax.set_ylabel('ones')
# grad_ax.set_xlabel('theta_1')
# grad_ax.set_zlabel('Cost')
# grad_fig.savefig(Path(__file__).parent / 'gradient_descent.png')

housing_data = pd.read_csv(Path(__file__).parent /'house_values.csv')
housing_lr = LinearRegression(X=housing_data[['sqft', 'bdrms']], Y=housing_data['value'])
housing_data['prediction'] = housing_lr.predict()
