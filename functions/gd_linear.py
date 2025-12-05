import numpy as np

def energy_function1(inputs, outputs, weights):
    gradient = (inputs.T@(inputs@weights-outputs))/(len(inputs))
    mse = (np.linalg.norm(inputs@weights - outputs)**2)/(2 * len(inputs))
    return gradient, mse

def gradient_descent1(training_data, testing_data, weights, step_size = 0.5, iterations = 1000):
    for i in range(iterations + 1):
        gradient_list, mse_train_list, mse_test_list = [], [], []
        for j in range(len(training_data)):
            gradient, mse = energy_function1(training_data[j][0], training_data[j][1], weights)
            gradient_list.append(gradient)
            mse_train_list.append(mse)
            mse_test = prediction_error(testing_data[j][0], testing_data[j][1], weights)
            mse_test_list.append(mse_test)

        gradient_mean = np.mean(gradient_list, axis = 0)
        mse_train_mean = np.mean(mse_train_list, axis = 0)
        mse_test_mean = np.mean(mse_test_list, axis = 0)
        weights = weights - step_size * gradient_mean

    final_mse_test_list = []
    for k in range(len(training_data)):
        final_mse_test = prediction_error(testing_data[k][0], testing_data[k][1], weights)
        final_mse_test_list.append(final_mse_test)
    final_mse_mean = np.mean(final_mse_test_list, axis = 0)

    return weights, final_mse_mean