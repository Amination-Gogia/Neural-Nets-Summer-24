## Coding a neuron, that is in our complex network
## Every neuron has a unique connection to each previous layer neuron

inputs = [1.2, 5.1, 2.1] ## outputs from neurons of the previous layer
weights = [3.1, 2.1, 8.7] ## weights associated to the connections
bias = 3

output = inputs[0] * weights[0] +inputs[1] * weights[1] + inputs[2] * weights[2] + bias

print(output)