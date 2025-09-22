# Linear regresion and Gradient Descent
# Visualizer: https://uclaacm.github.io/gradient-descent-visualiser/#playground

import numpy as np


### model training data needs to be provided by a real life problem statememt
def generate_sample_data_x_y_vectors() :

    # the straight line equation is  y = mx +c
    # lets take value of m = 3 and c =1   y = 3x + 1
    x_data = []
    y_data = []
    for i in range(10) :
        x = i
        y = 3*x + 1
        x_data.append(x)
        y_data.append(y)

    return x_data, y_data


def calculate_sum_array( arr ):
    return np.sum(arr)

def calculate_sum_square_array(input_arr):
    return np.sum(np.square(input_arr))


def calculate_xy_sum_array(x_data, y_data) :
    total = 0
    for x, y in zip(x_data, y_data):
        total += x * y

    return total


# Loss is mean square , function finds the derivative of  d(loss)/d(m)
def calculate_gradient_loss_with_m(m, c,  sum_of_x_square, sum_of_x_y, sum_of_x) :

    return ( 2*m*sum_of_x_square  - 2* sum_of_x_y +  2*c*sum_of_x      ) / 4



# Loss is mean square , function finds the derivative of  d(loss)/d(m)
def calculate_gradient_loss_with_c( m, c, sum_of_y, sum_of_x):
    return (2 * m * sum_of_x - 2 * sum_of_y + 8 * c ) / 4


def loss_func_calculation( m, c, np_x_array, np_y_array):

    diff_between_y_predicted = np.subtract ( np_y_array,   m*np_x_array  + c)
    #print(diff_between_y_predicted)
    return  np.sum(np.square ( diff_between_y_predicted)) / len(np_x_array)




if __name__ == '__main__':

    # generate Sample data
    x_data, y_data = generate_sample_data_x_y_vectors()
    np_x_array = np.array(x_data)
    np_y_array = np.array(y_data)
    print(np_x_array)
    print(np_y_array)

    m = 4
    c = 3
    learning_rate = 0.001
    epoch = 500

    # Values are only calculated once and reused during epoch iteration.
    sum_of_x_square = calculate_sum_square_array(np_x_array)
    sum_of_x_y = calculate_xy_sum_array(np_x_array, np_y_array)
    sum_of_x = calculate_sum_array(np_x_array)
    sum_of_y = calculate_sum_array(np_y_array)

    print("Based on artificial data value line equation is y = 3x + 1 , value of m= 3, c = 1")

    print("Before   model training parameters are m = {} , c = {} , learning_rate = {} , epoch = {} ".format(m,c, learning_rate, epoch))

    for epoch in range(epoch):
        new_m = m -   ( learning_rate *  calculate_gradient_loss_with_m( m, c,  sum_of_x_square, sum_of_x_y, sum_of_x) )
        new_c = c -  ( learning_rate * calculate_gradient_loss_with_c( m, c, sum_of_y, sum_of_x) )
        m = new_m
        c = new_c
        loss = loss_func_calculation(m, c, np_x_array, np_y_array)
        print( "m value is  {} c value is {} loss is {} ".format( m, c , loss))






