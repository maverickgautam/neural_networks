from nnfs.datasets import  spiral_data
import numpy as np
import matplotlib.pyplot as plot
import  nnfs

## neural network from scratch init
nnfs.init()

x, y = spiral_data(samples=200, classes=3)
plot.scatter(x[:,0],x[:,1],c=y, cmap='brg', edgecolors='k', s=50)

# Add labels and title
plot.xlabel('Feature 1')
plot.ylabel('Feature 2')
plot.title('Spiral Data Visualization')

plot.show()


class neural_network:


    # layer initialization
    def __init__(self, num_inputs, num_neurons):

        # Generate a 2D array with shape (num_neurons, num_inputs) containing random numbers
        self.weight =  .01 * np.random.randn(num_neurons, num_inputs)

        # Create a 2D array of zeros with 1 rows and num_neurons columns ,  broadcast of nump will be leveraged for addition 
        self.biases = np.zeros((1, num_neurons))



    def forward(self, inputs):

        self.output = inputs.dot( np.transpose(self.weight)) + self.biases

