import numpy as np

def huber_gradient(weights, paramater):
    gradient_weights = np.zeros(len(weights))
    for i in range(len(weights)):
        if weights[i] > paramater:
            gradient_weights[i] = 1
        elif abs(weights[i]) <= paramater:
            gradient_weights[i] = weights[i]/paramater
        elif weights[i] < -paramater:
            gradient_weights[i] = -1
    return gradient_weights

def energy_function3(inputs, outputs, weights, alpha, paramater):
    mse_part = (inputs.T@(inputs@weights-outputs))/(len(inputs))
    huber_loss_part = alpha * huber_gradient(weights, paramater)
    gradient = mse_part + huber_loss_part
    return gradient

def huber_loss_gradient_descent(training_data, testing_data, weights, alpha = 0.1, paramater = 0.5, step_size = 0.1, iterations = 1000):
    for i in range(iterations + 1):
        gradient_list, mse_train_list, mse_test_list = [], [], []
        for j in range(len(training_data)):
            gradient_list.append(energy_function3(training_data[j][0], training_data[j][1], weights, alpha, paramater))
            mse_train = prediction_error(training_data[j][0], training_data[j][1], weights)
            mse_train_list.append(mse_train)
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