import numpy as np
import matplotlib.pyplot as plt

def gradient_function(x):
    '''
        gradient of func_
    '''
    return 2*(x-2.5)

def func_(x):
    '''
        y = (x-2.5)^2 + 3
    '''
    return (x-2.5)**2 +3

def gradient_descent(start_x, lr, 
                     max_iterations=None,
                     epsilon=1e-5):
    '''
    perform gradient descent
    '''
    global history_x
    x = start_x
    gradient = gradient_function(x)
    i = 0
    while not np.all(np.absolute(gradient) <= epsilon):
        x = x - lr * gradient
        history_x.append(x)
        gradient = gradient_function(x)
        if np.any(np.absolute(gradient) > 1e7):
            print("error!!! can not converge!!!")
            print(gradient)
            exit(-1)

        i += 1
        if not max_iterations == None and i > max_iterations: 
            print("reach max iterations, stop!")
            break

    return x


if __name__ == "__main__":
    # prepare plot x y
    plot_x = np.linspace(-1, 6, 100)
    plot_y = func_(plot_x)
    plt.plot(plot_x, plot_y)
    # plt.show()

    # params
    start_x = 0.0
    learning_rate = 0.01
    epsilon = 1e-8
    max_iterations = 10000
    history_x = [start_x]

    # gradient descent
    final_x = gradient_descent(start_x, learning_rate, 
                               max_iterations = max_iterations, 
                               epsilon = epsilon)
    print(final_x)
    # plot x's trajactory
    plt.plot(np.array(history_x), func_(np.array(history_x)), 
             color="r", marker="*")
    plt.show()
