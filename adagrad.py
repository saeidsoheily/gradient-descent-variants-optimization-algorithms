__author__ = 'Saeid SOHILY-KHAH'
# Adagrad Python Implementation
# linear equation y = ax + b
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def func_plot(data, parameter):
    plt.plot(data[:,0], data[:,1], "x")
    x = np.r_[np.min(data[:,0])-5:np.max(data[:,0])+5:.1]
    plt.plot(x, parameter_updated[0] * x + parameter_updated[1],'-r')
    plt.title('AdaGrad')
    plt.text(65, 40, 'y = ax + b')
    plt.text(65, 35, 'a : ' + str(round(parameter[0],5)))
    plt.text(65, 30, 'b : ' + str(round(parameter[1],5)))
    plt.show()


def func_grad(data, parameter):
    """
    :param parameter:
    :param data:
    :param arg:
    :return: gradient of the loss function
    """
    #  Partial derivatives calcultion for a_gradient and b_gradient for error funcion f(X) = ((y_initial - y_predicted) ^ 2) / N
    grad = np.zeros(2)
    N = float(len(data))
    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]
        # (d)/(da)((y - (ax + b)) ^ 2 / N) = (2x (b + ax - y)) / N = -(2x / N (-b - ax + y)) = - (2x / N (y - (ax + b))
        grad[0] += - (2 / N) * x * (y - ((parameter[0] * x) + parameter[1]))
        # (d)/(db)((y - (ax + b)) ^ 2 / N) = (2 (b + ax - y)) / N = - (2 / N (-b - ax + y)) = - (2 / N (y - (ax + b))
        grad[1] += - (2 / N) * (y - ((parameter[0] * x) + parameter[1]))

    epsilon = 1e-6 # to ignore division by zero and nan values
    grad = np.divide(grad, N+ epsilon)
    return grad


def adagrad(data, parameter, func_grad, lr = 1e-2, epsilon = 1e-8, iterationNumber = 10000, *arg):
    """
    Adagrad Python Implementation
    Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. ML Research
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for adagrad (good default value is 0.01)
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    # Gti: a vector representing the diag(Gt) to store sum of the squares of the gradients up to time (iteration) t
    Gti = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        Gti += grad ** 2
        adjusted_grad = grad / (np.sqrt(Gti + epsilon))
        parameter = parameter - (lr * adjusted_grad)
    return parameter


if __name__ == '__main__':
    data = np.genfromtxt("data.csv", delimiter=" ")
    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adagrad algorithm
    parameter_updated = adagrad(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    func_plot(data, parameter_updated)
