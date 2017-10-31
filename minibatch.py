__author__ = 'Saeid SOHILY-KHAH'
# Mini-Batch Gradient Descent Python Implementation
# linear equation y = ax + b
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def func_plot(data, parameter):
    plt.plot(data[:,0], data[:,1], "x")
    x = np.r_[np.min(data[:,0])-5:np.max(data[:,0])+5:.1]
    plt.plot(x, parameter_updated[0] * x + parameter_updated[1],'-r')
    plt.title('Mini-Batch Gradient Descent')
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


def minibatch(data, parameter, func_grad, lr = 1e-2, minibatch_size = None, minibatch_ratio = 0.01, iterationNumber = 1000, *args):
    """
    Mini-batch gradient descent Python Implementation
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for mini-batch (good default value is 0.01)
    :param minibatch_size: if given is the number of samples considered in each iteration
    :param minibatch_ratio if minibatch_size is not set, it will be used to determine the batch size (depend on the length of the data)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    if minibatch_size is None:
        minibatch_size = int(math.ceil(len(data) * minibatch_ratio))
    for t in range(iterationNumber):
        sample_size = random.sample(range(len(data)), minibatch_size)
        np.random.shuffle(data)
        sample_data = data[0:sample_size[0], :]
        grad = func_grad(sample_data, parameter) # gradient of loss function
        parameter = parameter - (lr * grad)
    return parameter


if __name__ == '__main__':
    data = np.genfromtxt("data.csv", delimiter=" ")
    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # mini batch gradient descent algorithm
    parameter_updated = minibatch(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    func_plot(data, parameter_updated)
