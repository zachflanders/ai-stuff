
from pathlib import Path

import pandas

def hypothesis(X, theta):
    return theta[0] + X * theta[1]

def cost(X, Y, theta):
    return (1 / (2 * len(X))) * (hypothesis(X, theta) - Y).pow(2).sum()

def gradient_descent(X, Y, theta, alpha):
    return [
        theta[0] - (alpha / len(X)) * (hypothesis(X, theta) - Y).sum(),
        theta[1] - (alpha / len(X)) * ((hypothesis(X, theta) - Y) * X).sum(),
    ]

def predict(X, Y, theta, alpha, iterations):
    for i in range(iterations):        
        theta = gradient_descent(X, Y, theta, alpha)
    return hypothesis(X, theta), theta

def predict_one(population, theta):
    return pandas.Series([1, population / 1000]).dot(pandas.Series(theta)) * 10000

data = pandas.read_csv(Path(__file__).parent / 'food_truck_data.csv')
data['prediction'], final_theta = predict(X=data['pop'], Y=data['profit'], theta=[0, 0], alpha=0.01, iterations=1500)
scatter_plot = data.plot.scatter(x='pop', y='profit', marker='x', color='red')
data.plot.line(x='pop', y='prediction', ax=scatter_plot).get_figure().savefig('prediction.png')
print(predict_one(35000, final_theta))