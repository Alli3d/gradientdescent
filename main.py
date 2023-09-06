import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y):
    m_curr = b_curr = 0
    itr = 10000
    n = len(x)
    # the learning rate is a hyperparameter
    # you should try to find the ideal learning rate
    # (the one where the cost is decreasing the fastest)
    # cost can go up if the learning rate is poor
    # the cost will eventually stay the same if the rate is perfect
    learning_rate = 0.007

    for i in range(itr):

        y_pred = m_curr * x + b_curr

        # mse to calculate cost  - this is just a fancy way of writing it out
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])

        # m derivative and b derivative
        md = -(2/n)*sum(x*(y-y_pred))
        bd = -(2/n)*sum(y-y_pred)


        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if i % 100 == 0:
            print("cost {} m {}, b {}, itr {}".format(cost,m_curr,b_curr,i))

        if i % 1000 == 0:
            plt.plot(x, y_pred, color="blue")


x = np.array([5,6,7,8])
y = np.array([10,11,12,13])

gradient_descent(x,y)


plt.plot(x,y, color="red")
plt.show()