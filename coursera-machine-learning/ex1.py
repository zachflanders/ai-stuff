import pandas

data = pandas.read_csv('food_truck_data.csv')
scatter_plot = data.plot.scatter(x='pop', y='profit', marker='x', color='red')

alpha = 0.01
iterations = 1500
theta = [0, 0]

def hypothesis(X, theta):
    return theta[0] + X * theta[1]

def cost(X, Y, theta):
    return (1 / (2 * len(X))) * (hypothesis(X, theta) - Y).pow(2).sum()

def gradient_descent(X, Y, theta, alpha):
    return [
        theta[0] - (alpha / len(X)) * (hypothesis(X, theta) - Y).sum(),
        theta[1] - (alpha / len(X)) * ((hypothesis(X, theta) - Y) * X).sum(),
    ]

for i in range(iterations):
    if i % 100 == 0:
        print(cost(data['pop'], data['profit'], theta))
    theta = gradient_descent(data['pop'], data['profit'], theta, alpha)

data['prediction'] = hypothesis(data['pop'], theta)

data.plot.line(x='pop', y='prediction', ax=scatter_plot).get_figure().savefig('prediction.png')