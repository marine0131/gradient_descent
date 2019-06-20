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

def gradient_descent(X, y, init_theta, alpha):
    '''perform gradient descent
    '''
    theta = init_theta
    gradient = gradient_function(theta, X, y)
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha*gradient
        gradient = gradient_function(theta, X, y)
        if np.any(np.absolute(gradient) > 1e7):
            print("error!!! can not converge!!!")
            print(gradient)
            break;

    return theta


if __name__ == "__main__":
    # points with format (x1, x2, r)
    points = np.genfromtxt("beacon.csv", delimiter=",")
    X = []
    y = []
    # get trainin data
    # 2*(x1-x0)x + 2*(y1-y0)y = r0^2-r1^2-(x0^2-x1^2)-(y0^2-y1^2)
    for i in range(len(points)-1):
        xi = points[i, 0]
        xk = points[i+1, 0]
        yi = points[i, 1]
        yk = points[i+1, 1]
        ri = points[i, 2]
        rk = points[i+1, 2]
        x1 = 2 * (xk - xi)
        x2 = 2 * (yk - yi)
        X.append([x1, x2])
        y.append(ri*ri - rk*rk - (xi*xi-xk*xk) - (yi*yi - yk*yk))

    
    X = np.array(X)
    y = np.array(y).reshape(len(y),1)
    print X
    print y
    learning_rate = 0.0001
    init_theta = np.array([35,20]).reshape(2,1)
    optimal = gradient_descent(X, y, init_theta, learning_rate)
    print("optimal: ", optimal)

    if True:
        fig = plt.figure()
        axes = fig.add_subplot(111)
        for x0,y0,r in points:
            theta = np.arange(0, 2*np.pi, 0.01)
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            axes.plot(x, y)

        # draw optimal center
        axes.plot(optimal[0], optimal[1], "r+")
        plt.axis("equal")
        plt.show()

