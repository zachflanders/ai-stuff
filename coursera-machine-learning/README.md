# Machine Learning
## Standford University
### Taught by Andrew Ng
---
[Course Info](https://www.coursera.org/learn/machine-learning-course/home/info)

## Terms
- Hypothesis Function ($h_{\theta}$)
    - the model function that is attempting to represnt some real world relationship(s)
- Cost Function ($J$)
    - The function to be minimized in a optimization problem
- Gradient Descent
    - One general algorithm for minimizing a cost function
$$\text{repeat until convergence } \lbrace \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta) \rbrace$$

$$\alpha =  \text{learning rate}$$

$$\frac{\partial}{\partial \theta_j}J(\theta) = \text{The partial derivative of J with respect to }\theta_j$$ 


## Linear Regression
### Hypothesis Function
$$Y  = h_{\theta}(X)= \theta_o + \theta_1X$$
![Graph with scatter plot and linear regression](./ex1_linear_regression/prediction.png)
### Cost Function
$$J(\theta_0, \theta_1) = \frac{1}{2n}\sum_{i=1}^n(\hat{Y}_i - Y_i)^2$$
![Line chart of costs going down.](./ex1_linear_regression/costs_per_iteration.png)
### Gradient Descent
$$ \text{repeat until convergence } \lbrace 
    \\\\
    \theta_0 := \theta_0 - \alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x_i) - y_i)
    \\\\
    \theta_1 := \theta_1 - \alpha \frac{1}{m}\sum_{i=1}^m(h_{\theta}(x_i) - y_i)x_i
    \\\\
\rbrace$$
![Graph with gradient descent surface](./ex1_linear_regression/gradient_descent.png)