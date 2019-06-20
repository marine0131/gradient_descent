import numpy as np
import matplotlib.pyplot as plt

def error_function(theta, X, y):
    '''error function J definition
    '''
    diff = np.dot(X, theta) - y
    return (1./(2*len(y))) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    ''' gradient of the function J definition
    '''
    diff = np.dot(X, theta) - y
    return (1./len(y)) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, init_theta, alpha, max_iterations=None, error=1e-5):
    '''perform gradient descent
    '''
    theta = init_theta
    gradient = gradient_function(theta, X, y)
    i = 0
    while not np.all(np.absolute(gradient) <= error):
        theta = theta - alpha*gradient
        gradient = gradient_function(theta, X, y)
        if np.any(np.absolute(gradient) > 1e7):
            print("error!!! can not converge!!!")
            print(gradient)
            exit(-1)

        i += 1
        if not max_iterations==None and i > max_iterations: 
            print("reach max iterations, stop!")
            break;

    return theta


if __name__ == "__main__":
    # points with format (x,y)
    points = np.genfromtxt("data.csv", delimiter=",")

    X0 = np.ones((len(points), 1))
    X1 = points[:, 0].reshape(len(points), 1)
    X =  np.hstack((X0, X1))
    y = points[:, 1].reshape(len(points), 1)

    init_theta = np.array([0,0]).reshape(2,1)
    learning_rate = 0.000001
    error = 1
    max_iterations = 10000

    print("start gradient decent with error {}".format(error_function(init_theta, X, y)))
    print("Running ...")
    optimal_theta = gradient_descent(X, y, init_theta, learning_rate, 
            max_iterations=max_iterations, error=error)
    print("Optimal params: \n{}".format(optimal_theta))
    print("Error: {}".format(error_function(optimal_theta, X, y)))

    if True:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        # draw points
        axes.plot(X[:,1], y, "ro")
        # draw line
        xx = np.arange(10, 100, 0.01)
        yy = optimal_theta[0] + optimal_theta[1]*xx
        axes.plot(xx, yy)
        plt.axis("equal")
        plt.show()
