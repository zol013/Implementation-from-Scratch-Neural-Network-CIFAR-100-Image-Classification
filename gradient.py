import numpy as np
from neuralnet import Neuralnetwork

def check_grad(model, x_train, y_train):

    """
    TODO
        Checks if gradients computed numerically are within O(epsilon**2)

        args:
            model
            x_train: Small subset of the original train dataset
            y_train: Corresponding target labels of x_train

        Prints gradient difference of values calculated via numerical approximation and backprop implementation
    """
    epsilon = 10**(-2)
    results = []
    #1 output layer bias weight
    output_layer = model.layers[1]
    out_bias_w = output_layer.w[0][1]
    model(x_train, y_train)
    deltas = model.backward(False)
    delta = deltas[0][1]
    
    output_layer.w[0][1] = out_bias_w + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    output_layer.w[0][1] = out_bias_w - epsilon
    loss_minus, acc = model(x_train, y_train)
    output_layer.w[0][1] = out_bias_w
    
    num_delta = (loss_plus - loss_minus) / 2*epsilon

    abs_diff = abs(delta - num_delta)
    
    results.append(('output layer bias weight',num_delta, delta, abs_diff))
    
    #2 hidden bias weight
    hidden_layer = model.layers[0]
    hidden_bias_w = hidden_layer.w[0][1]
    model(x_train, y_train)
    deltas = model.backward(False)
    delta2 = deltas[0][1]
    
    hidden_layer.w[0][1] = hidden_bias_w + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    hidden_layer.w[0][1] = hidden_bias_w - epsilon
    loss_minus, acc = model(x_train, y_train)
    hidden_layer.w[0][1] = hidden_bias_w
    
    num_delta2 = (loss_plus - loss_minus) / 2*epsilon
    abs_diff2 = abs(delta2 - num_delta2)
    
    results.append(('hiden layer bias weight',num_delta2, delta2, abs_diff2))
    
    
    #3 hidden to output weight #1
    output_layer = model.layers[1]
    w3 = output_layer.w[4][0]
    
    model(x_train, y_train)
    deltas = model.backward(False)
    delta3 = deltas[4][0]
    
    output_layer.w[4][0] = w3 + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    output_layer.w[4][0] = w3 - epsilon
    loss_minus, acc = model(x_train, y_train)
    output_layer.w[4][0] = w3 

    num_delta3 = (loss_plus - loss_minus) / 2*epsilon
    abs_diff3 = abs(delta3 - num_delta3)
    
    results.append(('hidden to output weight #1', num_delta3, delta3, abs_diff3))
    
    
    #4 hidden to output weight #2
    output_layer = model.layers[1]
    w1 = output_layer.w[2][0]
    
    model(x_train, y_train)
    deltas = model.backward(False)
    delta4 = deltas[2][0]
    
    output_layer.w[2][0] = w1 + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    output_layer.w[2][0] = w1 - epsilon
    loss_minus, acc = model(x_train, y_train)
    output_layer.w[2][0] = w1 

    num_delta4 = (loss_plus - loss_minus) / 2*epsilon
    abs_diff4 = abs(delta4 - num_delta4)
    
    results.append(('hidden to output weight #2', num_delta4, delta4, abs_diff4))
    
    #5 input to hidden layer weight #1
    hidden_layer = model.layers[0]
    w1 = hidden_layer.w[2][0]
    
    model(x_train, y_train)
    deltas = model.backward(False)
    delta5 = deltas[2][0]
    
    hidden_layer.w[2][0] = w1 + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    hidden_layer.w[2][0] = w1 - epsilon
    loss_minus, acc = model(x_train, y_train)
    hidden_layer.w[2][0] = w1 

    num_delta5 = (loss_plus - loss_minus) / 2*epsilon
    abs_diff5 = abs(delta5 - num_delta5)
    results.append(('input to hidden weight #1', num_delta5, delta5, abs_diff5))
    
    
    #6 input to hidden layer weight #2
    hidden_layer = model.layers[0]
    w1 = hidden_layer.w[4][0]
    
    model(x_train, y_train)
    deltas = model.backward(False)
    delta6 = deltas[4][0]
    
    hidden_layer.w[4][0] = w1 + epsilon
    loss_plus, acc = model(x_train, y_train)
    
    hidden_layer.w[4][0] = w1 - epsilon
    loss_minus, acc = model(x_train, y_train)
    hidden_layer.w[4][0] = w1 

    num_delta6 = (loss_plus - loss_minus) / 2*epsilon
    abs_diff6 = abs(delta6 - num_delta6)
    results.append(('input to hidden weight #2', num_delta6, delta6, abs_diff6))
    
    return results



def checkGradient(x_train,y_train,config):

    subsetSize = 10  #Feel free to change this
    sample_idx = np.random.randint(0,len(x_train),subsetSize)
    x_train_sample, y_train_sample = x_train[sample_idx], y_train[sample_idx]

    model = Neuralnetwork(config)
    results = check_grad(model, x_train_sample, y_train_sample)
    return results