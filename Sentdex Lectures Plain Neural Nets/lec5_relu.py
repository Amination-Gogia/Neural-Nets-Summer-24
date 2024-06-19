import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

np.random.seed(0)
X = [[1, 2, 3, 2.5], [2, 5, -1, 2], [-1.5, 2.7, 3.3 , - 0.8]] ## a batch of outputs from neurons of the previous layer

inputs = [0, 2, 1, 3.3, 2.7, 1.1, 2.2, -100]
outputs = []

for i in inputs:
    # if i > 0:
    #     outputs.append(i)
    # elif i <= 0:
    #     outputs.append(0)
    outputs.append(max(0,i))

print(outputs)

X, y = spiral_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        ## n_inputs is number of neurons in previous layer
        ## n_neurons is number of neurons in this layer
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        ## sets the attribute 'output' to be the current weight * X + bias
        ## inputs can be a batch of training input features
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        ## Does the ReLU of the input array passed to it
        self.output = np.maximum(0, inputs)
    
layer1 = Layer_Dense(2, 5)

layer1.forward(X)

activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)