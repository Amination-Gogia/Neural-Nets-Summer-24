import numpy as np

def ReLU(inputs):
    return np.maximum(0, inputs)

def Softmax(inputs):
    ## inputs should be a matrix with columns = batch size, each column is an input vector
    max_val = np.max(inputs, keepdims= True, axis= 0)
    red_inp = inputs - max_val
    ## print(f'red: {red_inp}')
    exp_inp = np.exp(red_inp)
    denominator = np.sum(exp_inp, keepdims= True, axis= 0)
    norm_inputs = exp_inp/denominator
    return norm_inputs

def ReLU_derivative(inputs):
    return np.where(inputs > 0, 1, 0)

def Softmax_derivative():
    pass

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((n_neurons, 1))
    def forward(self, inputs):
        ## a_n = W_n-1 a_n-1 + b_n-1
        inputs = inputs.reshape(self.weights.shape[1], -1)
        # print(inputs.shape)
        self.output = (self.weights @ inputs + self.biases) ## a column vector
        

class Network:

    ## Input - HiddenLayers - OutputLayer->Softmax => Probability distribution
    def __init__(self, num_inputs, num_outputs, *hidden_layer_sizes):
        self.hidden_layers = [Layer(num_inputs, hidden_layer_sizes[0])]
        if len(hidden_layer_sizes) > 2:
            for i in range(1, len(hidden_layer_sizes)):
                self.hidden_layers.append(Layer(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]))
        elif len(hidden_layer_sizes) == 2:
            self.hidden_layers.append(Layer(hidden_layer_sizes[0], hidden_layer_sizes[1]))
        self.output_layer = Layer(hidden_layer_sizes[-1], num_outputs)
    
    def forward(self, inputs):
        self.hidden_layers[0].forward(inputs)
        # print(self.hidden_layers[0].output)
        if len(self.hidden_layers) >= 2:
            for i in range(1, len(self.hidden_layers)):
                relu_out = ReLU(self.hidden_layers[i - 1].output)
                self.hidden_layers[i].forward(relu_out)
                # print(self.hidden_layers[i].output.T)
        self.output_layer.forward(self.hidden_layers[-1].output)
        #self.result = Softmax(self.output_layer.output)
        self.result = Softmax(ReLU(self.output_layer.output)) ## Use for visualisation..., comment line 114 if you comment this, and uncomment the line before this.
    
    def compute_cost(self, X, y): 
        self.forward(X)
        y_hat = self.result
        y_hat = np.clip(y_hat, 10e-7, 1 - 10e-7)
        cost = 0
        if y.ndim == 1:
            ## Not one-hot coded
            ## y is expected to be a column vector with class numbers
            m = len(y)
            log_cost = np.log(y_hat)
            # print(log_cost)
            for i in range(m):
                cost += -1/m * (log_cost[y[i], i])
            #print(log_cost * y)
        else:
            ## y is a matrix, each column vector is a target probability vector
            m = y.shape[1]
            log_cost = np.log(y_hat)
            # for i in (tuple(y[i, :]).index(1) for i in range(m)):
            #     cost += -1/m * log_cost[i]
            cost = -1/m * np.sum(log_cost * y)
            # print(log_cost * y)
            # print(cost)
        return cost

class Trainer:
    def __init__(self, model, X, y):
        self.model = model
        self.target = y
        self.model.forward(X)
        self.hidd_w_diff_list = []
        self.hidd_b_diff_list = []
        self.a_diff_list = []
        for i in self.model.hidden_layers:
            self.hidd_w_diff_list.append(np.random.randn(*(i.weights.shape)))
            self.hidd_b_diff_list.append(np.random.randn(*(i.biases.shape)))
            self.a_diff_list.append(np.random.randn(*(i.biases.shape)))
        self.out_w_diff = np.random.randn(*(self.model.output_layer.weights.shape))
        self.out_b_diff = np.random.randn(*(self.model.output_layer.biases.shape))
        self.out_a_diff = np.random.randn(*(self.model.output_layer.biases.shape))
        self.X = X
        

    def calc_dc_da_outer(self):
        if self.target.ndim == 1:
            m = len(self.target)
            tar_class = self.target
        elif self.target.ndim > 1:
            m = self.target.shape[1]
            n = len(self.model.output_layer.biases)
            tar_class = np.zeros(m)
            temp = np.arange(n)
            for i in range(m):
                tar_class[i] = temp.dot(self.target[:, i])
                

        norm_output = self.model.result
        dc_da_outer = norm_output * ReLU_derivative(norm_output)
        dc_da_outer = 1/m * np.sum(norm_output, axis = 1, keepdims= True) ## can use np.average
        # print(tar_class)
        dc_da_outer[tar_class.astype(int), :] -= 1 
        self.out_a_diff = dc_da_outer 
        return dc_da_outer

    def calc_dc_da_hidden(self, layer_ind):
        n_hid_layers = len(self.model.hidden_layers)
        if layer_ind == (n_hid_layers - 1):
            # print(self.out_a_diff)
            # print(self.model.hidden_layers[-1].output)
            relu_a_out = ReLU_derivative(self.model.output_layer.output) * self.out_a_diff
            dc_da_last_hidd_matrix = self.model.output_layer.weights.T @ relu_a_out
            dc_da_last_hidd = np.average(dc_da_last_hidd_matrix, axis=1, keepdims= True)
            self.a_diff_list[layer_ind] = dc_da_last_hidd
            return dc_da_last_hidd
        elif layer_ind < (n_hid_layers - 1):
            relu_a_out = ReLU_derivative(self.model.hidden_layers[layer_ind + 1].output) * self.a_diff_list[layer_ind + 1].reshape(-1, 1)
            # print(self.model.hidden_layers[layer_ind + 1])
            dc_da_curr_matrix = self.model.hidden_layers[layer_ind + 1].weights.T @ relu_a_out
            dc_da_curr = np.average(dc_da_curr_matrix, axis=1, keepdims= True)
            self.a_diff_list[layer_ind] = dc_da_curr
            return dc_da_curr
        # print(dc_da_last_hidd)
        pass

    def calc_dc_db_hidden(self, layer_ind):
        n_hid_layers = len(self.model.hidden_layers)
        if layer_ind < n_hid_layers - 1:
            self.hidd_b_diff_list[layer_ind] = np.average(ReLU_derivative(self.model.hidden_layers[layer_ind].output) * (self.a_diff_list[layer_ind]), axis=1, keepdims=True)
            return self.hidd_b_diff_list[layer_ind]
        elif layer_ind == n_hid_layers - 1:
            self.hidd_b_diff_list[layer_ind] = np.average(ReLU_derivative(self.model.hidden_layers[layer_ind].output) * (self.a_diff_list[layer_ind]), axis=1, keepdims= True)
            return self.hidd_b_diff_list[layer_ind]
    
    def calc_dc_dw_hidden(self, layer_ind):
        n_hid_layers = len(self.model.hidden_layers)
        if layer_ind == 0:
            temp = ReLU_derivative(self.model.hidden_layers[layer_ind].output) * self.a_diff_list[0]
            ## print(f'temp: {temp}')
            self.hidd_w_diff_list[0] = temp @ (self.X.T) 
            return self.hidd_w_diff_list[0]
        elif layer_ind <= n_hid_layers:
            temp = ReLU_derivative(self.model.hidden_layers[layer_ind].output) * self.a_diff_list[layer_ind]
            self.hidd_w_diff_list[layer_ind] = temp @ (self.model.hidden_layers[layer_ind - 1].output).T 
            ## print(f'Check pls: {temp @ self.X.T}')
            return self.hidd_w_diff_list[layer_ind]
    def calc_dc_dw_outer(self):
        prev_a_sum = np.sum(self.model.hidden_layers[-1].output, axis= 1, keepdims=True)
        # print(prev_a_avg)
        dc_dw_outer =  self.out_a_diff @ prev_a_sum.T
        self.out_w_diff = dc_dw_outer 
        return dc_dw_outer

    def calc_dc_db_outer(self):
        self.out_b_diff = self.out_a_diff
        return self.out_a_diff
    
    def lower_the_cost(self, iter, alpha):
        for i in range(iter):
            t.model.output_layer.weights -= (t.out_w_diff * alpha)
            t.model.output_layer.biases -= (t.out_b_diff * alpha)
            t.model.hidden_layers[0].weights -= (alpha * t.hidd_w_diff_list[0])
            t.model.forward(self.X)
            n = len(self.model.hidden_layers)
            self.calc_dc_da_outer()
            self.calc_dc_db_outer()
            self.calc_dc_dw_outer()
            self.model.output_layer.weights -= self.out_w_diff * alpha
            self.model.output_layer.biases -= self.out_b_diff * alpha
            for j in range(n):
                self.calc_dc_da_hidden(n - j - 1)
                self.calc_dc_db_hidden(n - j - 1)
                self.calc_dc_dw_hidden(n - j - 1)
                self.model.hidden_layers[n - j - 1].weights -= self.hidd_w_diff_list[n - j - 1] * alpha
                self.model.hidden_layers[n - j - 1].biases -= self.hidd_b_diff_list[n - j - 1] * alpha
            # print(t.model.result)
            # print(t.model.compute_cost(X_trial, y_trial))

# l1 = Layer(2,3)


n1 = Network(3,3,4)

# n1.forward(np.array(([1,2,3], [2,34,4])))
# print(n1.result)

# c = n1.compute_cost(np.array(([1,2,3])), np.array([1]))
# print(c)

X_trial = np.array(([1,2,3], [4,5,6])).T
y_trial = np.array(([1,0,0], [0,1,0])).T

t = Trainer(n1, X_trial, y_trial)
print(t.model.result)
print(t.model.compute_cost(X_trial, y_trial))
# print(t.calc_dc_da_outer())

alpha = 0.001
# print(t.model.output_layer.biases)
# t.model.output_layer.weights - t.out_w_diff * alpha
# t.model.output_layer.biases -= t.out_b_diff * alpha
# print(t.model.output_layer.biases)
# t.model.forward(X_trial)
# t.calc_dc_da_outer()
# t.calc_dc_db_outer()
# t.calc_dc_dw_outer()
# # print(t.model.result)
# print(t.model.compute_cost(X_trial, y_trial))
# # print(t.calc_dc_da_outer())
# print(t.model.hidden_layers)

# print(t.calc_dc_da_hidden(0))
# print(t.model.hidden_layers[1].biases)
# print(t.calc_dc_db_hidden(0))
# print(t.hidd_b_diff_list[0])
# for i in range(100):
#     t.calc_dc_da_outer()
#     t.calc_dc_db_outer()
#     t.calc_dc_dw_outer()
#     t.calc_dc_da_hidden(1)
#     t.calc_dc_dw_hidden(1)
#     t.calc_dc_db_hidden(1)
#     t.calc_dc_da_hidden(0)
#     t.calc_dc_dw_hidden(0)
#     t.calc_dc_db_hidden(0)
#     t.model.output_layer.weights -= (t.out_w_diff * alpha)
#     t.model.output_layer.biases -= (t.out_b_diff * alpha)
#     # print(t.hidd_w_diff_list[0])
#     # # print(t.model.hidden_layers[0].weights)
#     # print(t.hidd_w_diff_list[1])
#     t.model.hidden_layers[0].weights -= (alpha * t.hidd_w_diff_list[0])
#     t.model.hidden_layers[1].weights -= (alpha * t.hidd_w_diff_list[1])
#     t.model.hidden_layers[0].biases -= (alpha * t.hidd_b_diff_list[0])
#     t.model.hidden_layers[1].biases -= (alpha * t.hidd_b_diff_list[1])
#     t.model.forward(X_trial)
    
    # print(t.model.hidden_layers[1].biases)
    
    
    
t.lower_the_cost(100, 0.001)
    
print(t.model.result)
print(t.model.compute_cost(X_trial, y_trial))