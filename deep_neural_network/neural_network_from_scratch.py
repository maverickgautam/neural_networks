# https://www.youtube.com/watch?v=sPlBSyJeDqA&list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu&index=21

from nnfs.datasets import  spiral_data
import numpy as np
import matplotlib.pyplot as plot
import  nnfs

class Util:

    def generate_random_data(self) :
        ## neural network from scratch init
        nnfs.init()
        x, y = spiral_data(samples=200, classes=3)
        plot.scatter(x[:,0],x[:,1],c=y, cmap='brg', edgecolors='k', s=50)

        # Add labels and title
        plot.xlabel('Feature 1')
        plot.ylabel('Feature 2')
        plot.title('Spiral Data Visualization')

        plot.show()

        self.x = x
        self.y = y



class Relu:

    def forward(self, inputs):

        self.inputs  = inputs
        self.output  = np.maximum(0, inputs)

    def backward(self, dvalues):

        self.dinputs  = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:

    #Transforms raw outputs (logits) into a probability distribution.
    #Makes the largest value closer to 1, and the rest smaller — but still positive and sum to 1.
    def forward(self, inputs):

        # axis = 1 means for a given row take sum across columns.
        # softmax is calculated across neurons
        # inputs is expected to have number of columns = number of nuerons in the softmax layer
        exp_values     = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) # improve stability
        probabilities  = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output    = probabilities


class Loss:

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)

        data_loss = np.mean(sample_losses)

        return data_loss



class Loss_CategoricalCrossentropy(Loss):


    def forward(self, y_pred, y_true):
        # no of sampled in a batch
        samples = len(y_pred)

        #Clips all values in array so that:
            # Any value less than min_value becomes min_value
            # Any value greater than max_value becomes max_value

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # .shape retruns a single value for a 1D array and 2 values (a,b) for 2D array
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true ]


        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)


        negative_log_likelikelyhood = -np.log(correct_confidence)
        return negative_log_likelikelyhood

    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        labels = len(dvalues[0])

        # turn into one hot encoding
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # calculate the gradient
        self.dinputs = -y_true/dvalues
        # normalize the gradient
        self.dinputs = self.dinputs/samples



class Activation_Softmax_Loss_CategoricalCrossentropy:

    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossentropy()

    def forward(self, inputs, y_true):

        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate( self.output, y_true)


    def backward(self, dvalues, y_true):

        samples = len(dvalues)

        # If labels are one hot encoded turn them to discrete value
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        #calculate Gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize Gradient
        self.dinputs = self.dinputs/samples






# inputs of one batch of data is represented as rows [x1, x2, x3, x4 ] -> one batch of data example. Multiple batch means multiple rows
# weights will be constructed with neurons as no of column.
class Neural_Layer:

    def __init__(self, n_inputs, n_neurons ):
        super().__init__()
        # this step ensures we dont have to take Transpose of Weights.
        # To have a clear mental model its  n_neurons * n_inputs where n_neurons is number of
        # neurons in the layer and n_inputs is weights from the inputs
        self.weights = np.random.randn(n_inputs, n_neurons)
        # if you take one row of input and and weights  you will end up with one row in output
        self.biases = np.zeros((1, n_neurons))


    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights ) + self.biases

    # dvalues  is dl_dz in https://www.youtube.com/watch?v=sPlBSyJeDqA&list=PLPTV0NXA_ZSj6tNyn_UadmUeU3Q3oR-hu&index=21
    # dvalues -> dl_dz is matrix with number of neurons as columns.  and there can be batched of inputs and hence batches of  DL/DZbatch
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        # go to columns and sum down rows.  Loss = L1 + L2 + L3 for different batches of Input DLoss/Dbias = L1/bias1 + L2/Bias1 + L3/Bias1
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class optimizer :
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_parameters(self, layer:Neural_Layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

if __name__ == '__main__':

    util = Util()

    #Create Data Set
    util.generate_random_data()
    optimizer = optimizer(learning_rate=0.1)

    dense1 = Neural_Layer(2, 3)
    reulu_activation  = Relu()
    dense2 = Neural_Layer(3, 3)
    loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()


    for epoch in range(1000):
        # forward pass
        dense1.forward(util.x)
        reulu_activation.forward(dense1.outputs)
        dense2.forward(reulu_activation.output)
        loss = loss_activation.forward(dense2.outputs, util.y)

        #print(loss_activation.output[:5])
        print("loss is  : " , loss)

        prediction = np.argmax(loss_activation.output, axis=1)
        if len(util.y.shape) == 2:
            y = np.argmax(util.y, axis=1)


        accuracy = np.mean(prediction == util.y)

        print('accuracy ' , accuracy)

        # Backward pass
        loss_activation.backward(loss_activation.output, util.y)
        dense2.backward(loss_activation.dinputs)
        reulu_activation.backward(dense2.dinputs)
        dense1.backward(reulu_activation.dinputs)

        optimizer.update_parameters(dense1)
        optimizer.update_parameters(dense2)



    print(dense1.weights)
    print(dense1.biases)
    print(dense2.weights)
    print(dense2.biases)










