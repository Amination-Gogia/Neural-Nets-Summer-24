import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
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

class Activation_Softmax:
    def forward(self, inputs):
        max_val = np.max(inputs, axis = 1, keepdims=True)
        modified_inputs = inputs - max_val
        exp_values = np.exp(modified_inputs)
        norm_values = exp_values/np.sum(exp_values, axis = 1, keepdims= True) ## Probabilities
        self.output = norm_values

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                           [0.1, 0.5, 0.4],
                           [0.02, 0.9, 0.08]])

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis= 1)
        
        negative_log_likelihoods = - np.log(correct_confidences)
        return negative_log_likelihoods

class back_prop:
    # def back_prop(self, network, X, y, y_hat, alpha):
    #     if y.shape
    pass

class Network:
    def __init__(self, *args):
        # self.layers = []
        # if all(isinstance(x, int) for x in args):
        #     for(x in )
        pass

class_targets = [0, 1, 1]
# neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
# average_loss = np.mean(neg_log)

# print(average_loss)
X, y = spiral_data(samples= 100, classes= 3)


dense1 = Layer_Dense(2, 3)

dense1.forward(X)

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
loss_function = Loss_CategoricalCrossentropy()

loss = loss_function.calculate(activation2.output, y)

predictions = np.argmax(softmax_outputs, axis = 1)

accuracy = np.mean(predictions == class_targets)

print('acc: ' + str(accuracy))