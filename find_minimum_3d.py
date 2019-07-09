'''
    find global valley of a surface using gradient descent
'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_surface(x, y, z):
    # ax = plt.gca(projection='3d')
    fig = plt.figure()
    ax3d = Axes3D(fig)
    plt.title('3D surface', fontsize=20)
    ax3d.set_xlabel('x', fontsize=14)
    ax3d.set_ylabel('y', fontsize=14)
    ax3d.set_zlabel('z', fontsize=14)
    ax3d.set_alpha(0.5)
    plt.tick_params(labelsize=10)
    ax3d.plot_surface(x, y, z, rstride=10, cstride=10, cmap="jet")
    # plt.show()
    return ax3d

def gradient_function(x):
    dx0 = np.exp(-x[0]**2-x[1]**2) * ((-1/2 + 5*x[0]**4) +
                                      (1-x[0]/2+x[0]**5+x[1]**3)*(-2*x[0]))
    dx1 = np.exp(-x[0]**2-x[1]**2) * (3*x[1]**2 +
                                      (1-x[0]/2+x[0]**5+x[1]**3)*(-2*x[1]))
    return np.array([dx0, dx1])

def func_(x, y):
    '''
        function 
        x: [nparray] N*M, N is number of sample, M is dimension
    '''
    return (1 - x/2 + x**5 + y**3) * np.exp(- x**2 - y**2)

def gradient_descent(start_x, lr, 
                     max_iterations=None, epsilon=1e-5):
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
    plot_x, plot_y = np.meshgrid(np.linspace(-3, 3, 1000), np.linspace(-3, 3, 1000))
    plot_z =  func_(plot_x, plot_y)

    ax = plot_surface(plot_x, plot_y, plot_z)

    # params
    start_x = np.array([-0.5, -0.5])  # try different start will get different result
    learning_rate = 0.01
    epsilon = 1e-8
    max_iterations = 10000

    history_x = [start_x]

    final_x = gradient_descent(start_x, learning_rate, 
                               max_iterations = max_iterations, 
                               epsilon = epsilon)
    print("final x: {}, minimum value {}".format(final_x, func_(final_x[0], final_x[1])))

    history_x = np.array(history_x)
    ax.plot(history_x[:, 0], history_x[:, 1], 
            func_(history_x[:,0], history_x[:,1]), 
            c='w')
            
    plt.show()


