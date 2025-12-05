import numpy as np

def prediction_error(inputs, outputs, weights):
    return (np.linalg.norm(inputs@weights - outputs)**2)/(2 * len(inputs))

def linear_data(data):
    intercept = np.ones((len(data), 1))
    return np.c_[intercept, data]

def linear_regression(inputs, outputs):
    return np.linalg.solve(inputs.T@inputs, inputs.T@outputs)

def linear_analysis(training_data, testing_data):
    weights_list, mse_list = [], []

    for i in range(len(training_data)):
        weights = linear_regression(training_data[i][0], training_data[i][1])
        weights_list.append(weights)
        mse = prediction_error(testing_data[i][0], testing_data[i][1], weights)
        mse_list.append(mse)

    weights_mean = np.mean(weights_list, axis = 0)
    weights_std = np.std(weights_list, axis = 0)
    mse_mean = np.mean(mse_list, axis = 0)
    mse_std = np.std(mse_list, axis = 0)

    return weights_mean, weights_std, mse_mean, mse_std