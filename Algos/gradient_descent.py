import numpy as np;

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    n = len(x)
    Iteration = 10000
    learning_rate = 0.001

    for i in range(Iteration):
        y_predicted = m_curr * x + b_curr                           # Simple Linear Method prediction to find "Y"

        meanSquareError =  (1/n)  * sum ( [i**2 for i in (y - y_predicted)] )       #AKA cost function

        m_derivative = -(2/n) * sum( x * (y - y_predicted))         # M-derivative
        b_derivative = -(2/n) * sum(y - y_predicted)                # b-derivative

        m_curr = m_curr - learning_rate * m_derivative              # learning rate = amount of change | 
        b_curr = b_curr - learning_rate * b_derivative              # learning rate = amount of change | 

        print("m {}, b {}, Cost {}, i {}".format(m_curr, b_curr, meanSquareError, i ))


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)
