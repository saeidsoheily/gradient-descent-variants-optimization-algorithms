__author__ = 'Saeid SOHILY-KHAH'
# Gradient Descent Variants Optimization Algorithms
# linear equation y = ax + b
import math
import random
import numpy as np
from matplotlib import pyplot as plt


def func_plot(data, parameter, title):
    plt.plot(data[:,0], data[:,1], "x")
    x = np.r_[np.min(data[:,0])-5:np.max(data[:,0])+5:.1]
    plt.plot(x, parameter_updated[0] * x + parameter_updated[1],'-r')
    # plt.axis([np.min(data[:,0])-10, np.max(data[:,0])+10, np.min(data[:,1])-10, np.max(data[:,1])+10])
    plt.title(title)
    plt.text(65, 40, 'y = ax + b')
    plt.text(65, 35, 'a : ' + str(round(parameter[0],5)))
    plt.text(65, 30, 'b : ' + str(round(parameter[1],5)))


def func_computeError(data, parameter):
    # linear equation y = ax + b
    totalError = 0
    for i in range(0, len(data)):
        x = data[i, 0]
        y = data[i, 1]
        totalError += (y - (parameter[0] * x + parameter[1])) ** 2
    return totalError / float(len(data))


def func_plot_totalErrors():
    global totalError
    titles = ['Mini-Batch', 'Momentum', 'NaG', 'AdaGrad', 'AdaDelta','RMSprop', 'Adam', 'AdaMax', 'Nadam']
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.plot(totalError[i]/sum(totalError[i]))
        plt.title(titles[i])
    plt.show()


def func_grad(data, parameter):
    """
    :param parameter:
    :param data:
    :param arg:
    :return: gradient of the loss function (sample function)
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
    grad = np.divide(grad, N + epsilon)
    return grad


def minibatch(data, parameter, func_grad, lr = 1e-2, minibatch_size = None, minibatch_ratio = 0.01, iterationNumber = 500, *args):
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
    global totalError
    if minibatch_size is None:
        minibatch_size = int(math.ceil(len(data) * minibatch_ratio))
    for t in range(iterationNumber):
        sample_size = random.sample(range(len(data)), minibatch_size)
        np.random.shuffle(data)
        sample_data = data[0:sample_size[0], :]
        grad = func_grad(sample_data, parameter) # gradient of loss function
        parameter = parameter - (lr * grad)
        totalError[0].append(func_computeError(data, parameter))
    return parameter


def momentum(data, parameter, func_grad, lr = 1e-2, gamma = 0.9, iterationNumber = 500, *arg):
    """
    Momentum Python Implementation
    Qian, N. (1999). On the momentum term in gradient descent learning algorithms. Neural Networks
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for momentum (good default value is 0.01)
    :param gamma: momentum term
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    global totalError
    vt = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        vt = gamma * vt + lr * grad
        parameter = parameter - vt
        totalError[1].append(func_computeError(data, parameter))
    return parameter


def nag(data, parameter, func_grad, lr = 1e-2, gamma = 0.9, iterationNumber = 500, *arg):
    """
    Nesterov accelerated gradient Python Implementation
    Nesterov, Y. (1983). A method for unconstrained convex minimization problem with the rate of convergence o(1/k2).
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for nag (good default value is 0.01)
    :param gamma: NaG term
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    global totalError
    vt = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter - gamma * vt) # gradient of loss function
        vt = gamma * vt + lr * grad
        parameter = parameter - vt
        totalError[2].append(func_computeError(data, parameter))
    return parameter


def adagrad(data, parameter, func_grad, lr = 1e-1, epsilon = 1e-8, iterationNumber = 500, *arg):
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
    global totalError
    # Gti: a vector representing the diag(Gt) to store sum of the squares of the gradients up to time (iteration) t
    Gti = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        Gti += grad ** 2
        adjusted_grad = grad / (np.sqrt(Gti + epsilon))
        parameter = parameter - (lr * adjusted_grad)
        totalError[3].append(func_computeError(data, parameter))
    return parameter


def adadelta(data, parameter, func_grad, rho = 0.9, epsilon = 1e-5, iterationNumber = 500, *arg):
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
    global totalError
    E_grad2 = np.zeros(parameter.shape[0])
    E_delta_para2 = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        E_grad2 = (rho * E_grad2) + ((1. - rho) * (grad ** 2))
        delta_parameters = - (np.sqrt(E_delta_para2 + epsilon)) / (np.sqrt(E_grad2 + epsilon)) * grad
        E_delta_para2 = (rho * E_delta_para2) + ((1. - rho) * (delta_parameters ** 2))
        parameter += delta_parameters
        totalError[4].append(func_computeError(data, parameter))
    return parameter


def rmsprop(data, parameter, func_grad, lr = 1e-2, rho = 0.9, epsilon = 1e-6, iterationNumber = 500, *arg):
    """
    RMSprop implementation
    adaptive learning rate method proposed by Geoff Hinton
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for rmsprop (good default value is 0.01)
    :param rho: the global factor for adadelta (good default value is 0.9)
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    global totalError
    E_grad2 = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        E_grad2 = (rho * E_grad2) + ((1. - rho) * (grad ** 2))
        parameter = parameter - (lr / (np.sqrt(E_grad2 + epsilon)) * grad)
        totalError[5].append(func_computeError(data, parameter))
    return parameter


def adam(data, parameter, func_grad, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, iterationNumber = 500, *arg):
    """
    Adam implementation
    Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations,
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for adam (good default value is 0.01)
    :param beta1
    :param beta2
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    global totalError
    mt = np.zeros(parameter.shape[0])
    vt = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        mt = beta1 * mt + (1. - beta1) * grad
        vt = beta2 * vt + (1. - beta2) * grad ** 2
        mt_hat = mt / (1. - beta1 ** (t+1))
        vt_hat = vt / (1. - beta2 ** (t+1))
        parameter = parameter - (lr / (np.sqrt(vt_hat) + epsilon)) * mt_hat
        totalError[6].append(func_computeError(data, parameter))
    return parameter


def adamax(data, parameter, func_grad, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, iterationNumber = 500, *arg):
    """
    Adamax implementation
    Kingma, D. P., & Ba, J. L. (2015). Adam: a Method for Stochastic Optimization. International Conference on Learning Representations,
    :param data: the data
    :param parameter: the start point for the optimization
    :param func_grad: returns the loss functions gradient
    :param lr: the global learning rate for adam (good default value is 0.01)
    :param beta1
    :param beta2
    :param epsilon: a small number to counter numerical instabiltiy (e.g. zero division)
    :param iterationNumber: the number of iterations which algorithm will run
    :param *args: a list or tuple of additional arguments (e.g. passed to function func_grad)
    :return:
    """
    global totalError
    mt = np.zeros(parameter.shape[0])
    ut = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        mt = beta1 * mt + (1. - beta1) * grad
        ut = np.maximum(beta2 * ut, np.abs(grad))
        mt_hat = mt / (1. - beta1 ** (t+1))
        parameter = parameter - ((lr / (ut + epsilon)) * mt_hat)
        totalError[7].append(func_computeError(data, parameter))
    return parameter


def nadam(data, parameter, func_grad, lr = 1e-2, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-6, iterationNumber = 500, *arg):
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
    global totalError
    mt = np.zeros(parameter.shape[0])
    vt = np.zeros(parameter.shape[0])
    for t in range(iterationNumber):
        grad = func_grad(data, parameter) # gradient of loss function
        mt = beta1 * mt + (1. - beta1) * grad
        vt = beta2 * vt + (1. - beta2) * grad ** 2
        mt_hat = mt / (1. - beta1 ** (t+1))
        vt_hat = vt / (1. - beta2 ** (t+1))
        parameter = parameter - (lr / (np.sqrt(vt_hat) + epsilon) * (beta1 * mt_hat + ((1 - beta1) * grad / (1 - beta1 ** (t+1)))))
        totalError[8].append(func_computeError(data, parameter))
    return parameter

if __name__ == '__main__':
    # to save the computed total errorin each method
    global totalError
    totalError =  [[] for _ in range(9)]

    # read the sample data
    data = np.genfromtxt("data.csv", delimiter=" ")

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # mini batch gradient descent algorithm
    parameter_updated = minibatch(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 1)
    func_plot(data, parameter_updated,'Mini-Batch')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adagrad algorithm
    parameter_updated = momentum(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 2)
    func_plot(data, parameter_updated,'Momentum')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adagrad algorithm
    parameter_updated = nag(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 3)
    func_plot(data, parameter_updated,'NaG')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adagrad algorithm
    parameter_updated = adagrad(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 4)
    func_plot(data, parameter_updated,'AdaGrad')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adadelta algorithm
    parameter_updated = adadelta(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 5)
    func_plot(data, parameter_updated,'AdaDelta')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # rmsprop algorithm
    parameter_updated = rmsprop(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 6)
    func_plot(data, parameter_updated,'RMSprop')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adam algorithm
    parameter_updated = adam(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 7)
    func_plot(data, parameter_updated,'Adam')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # adamax algorithm
    parameter_updated = adamax(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 8)
    func_plot(data, parameter_updated,'AdaMax')

    parameter = np.zeros(2) # the start point for the optimization   parameter = [p1 p2 ... pn]
    # nadam algorithm
    parameter_updated = nadam(data, parameter, func_grad)
    # plot the data and y = ax + b with the updated parameters
    plt.subplot(3, 3, 9)
    func_plot(data, parameter_updated,'Nadam')

    plt.show()

    func_plot_totalErrors()
