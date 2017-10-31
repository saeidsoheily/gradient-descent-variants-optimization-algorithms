__author__ = 'Saeid SOHILY-KHAH'
# Nadam Python Implementation
# linear equation y = ax + b
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def func_plot(data, parameter):
    plt.plot(data[:,0], data[:,1], "x")
    x = np.r_[np.min(data[:,0])-5:np.max(data[:,0])+5:.1]
    plt.plot(x, parameter_updated[0] * x + parameter_updated[1],'-r')
    plt.title('Nadam')
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


def nadam(data, parameter, func_grad, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, iterationNumber = 1000, *arg):
    """
    Nadam implementation
    Dozat, T. (2016). Incorporating Nesterov Momentum into Adam. ICLR Workshop, (1), 2013–2016.
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for nadam (good default value is 0.01)
    :param beta1
    :param beta2
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    mt = np.zeros(parameter.shape[0])
    vt = np.zeros(parameter.shape[0])

    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        mt = beta1 * mt + (1. - beta1) * grad
        vt = beta2 * vt + (1. - beta2) * grad ** 2
        mt_hat = mt / (1. - beta1 ** (t+1))
        vt_hat = vt / (1. - beta2 ** (t+1))
        parameter = parameter - (lr / (np.sqrt(vt_hat) + epsilon) * (beta1 * mt_hat + ((1 - beta1) * grad / (1 - beta1 ** (t+1)))))
    return parameter


if __name__ == '__main__':
    data = np.genfromtxt("data.csv", delimiter=" ")
    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # nadam algorithm
    parameter_updated = nadam(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    func_plot(data, parameter_updated)