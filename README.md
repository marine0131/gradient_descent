examples demonstrates how gradient descent work in linear regression, beacon location ...

## linear regression  
data.csv is a 2 params linear regression data, download from https://github.com/mattnedrich/GradientDescentExample

![image](https://github.com/marine0131/gradient_descent/blob/master/pic/linear_regression.png)


## beacon location
beacon.csv is beacon location data, data format is as below:

    col 1: x of beacon n; 
    col 2: y of beacon n; 
    col 3: measured distance between current pose and neacon n

adjust distances in beacon.csv to see the result

![image](https://github.com/marine0131/gradient_descent/blob/master/pic/beacon.png)

## find_minimum of a curve
find_minimum_2d.py

adjust learning_rate, see how the learning_rate affect the coverge steps

![image](https://github.com/marine0131/gradient_descent/blob/master/pic/curve.png)

## find_minimum of a surface
find_minimum_3d.py

try to adjust start [x,y], find how start point affect the result(easily converge to local optimum)

![image](https://github.com/marine0131/gradient_descent/blob/master/pic/surface.png)
