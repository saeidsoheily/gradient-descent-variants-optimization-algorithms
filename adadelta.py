__author__ = 'Saeid SOHILY-KHAH'
# Adadelta Python Implementation
# linear equation y = ax + b
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def func_plot(data, parameter):
    plt.plot(data[:,0], data[:,1], "x")
    x = np.r_[np.min(data[:,0])-5:np.max(data[:,0])+5:.1]
    plt.plot(x, parameter_updated[0] * x + parameter_updated[1],'-r')
    plt.title('AdaDelta')
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


def adadelta(data, parameter, func_grad, rho = 0.9, epsilon = 1e-5, iterationNumber = 1000, *arg):
    """
    Adadelta implementation
    Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method.
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param rho: the global factor for adadelta (good default value is 0.9)
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    E_grad2 = np.zeros(parameter.shape[0])
    E_delta_para2 = np.zeros(parameter.shape[0])

    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        E_grad2 = (rho * E_grad2) + ((1. - rho) * (grad ** 2))
        delta_parameters = - (np.sqrt(E_delta_para2 + epsilon)) / (np.sqrt(E_grad2 + epsilon)) * grad
        E_delta_para2 = (rho * E_delta_para2) + ((1. - rho) * (delta_parameters ** 2))
        parameter += delta_parameters
    return parameter


if __name__ == '__main__':
    data = np.genfromtxt("data.csv", delimiter=" ")
    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adadelta algorithm
    parameter_updated = adadelta(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    func_plot(data, parameter_updated)

