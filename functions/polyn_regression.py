import numpy as np

def poly_data(data, degree):
    x_matrix = np.ones((len(data), 1))
    for i in range(1, degree + 1):
        x_matrix = np.c_[x_matrix, np.power(data, i)]
    return x_matrix

def poly_regression(inputs, outputs):
    return np.linalg.solve(inputs.T@inputs, inputs.T@outputs)

def poly_analysis(training_data, testing_data, degree):
    for i in range(1, degree+1):
        poly_training_data, poly_testing_data = [], []
        for j in range(len(testing_data)):
            poly_training_data.append((poly_data(training_data[j][0], i), training_data[j][1]))
            poly_testing_data.append((poly_data(testing_data[j][0], i), testing_data[j][1]))

        weights_list, mse_list = [], []
        for k in range(len(poly_training_data)):
            weights = poly_regression(poly_training_data[k][0], poly_training_data[k][1])
            weights_list.append(weights)
            mse = prediction_error(poly_testing_data[k][0], poly_testing_data[k][1], weights)
            mse_list.append(mse)

        weights_mean = np.mean(weights_list, axis = 0)
        weights_std = np.std(weights_list, axis = 0)
        mse_mean = np.mean(mse_list, axis = 0)
        mse_std = np.std(mse_list, axis = 0)

        if i == 1:
            best_weights_mean, best_weights_std = weights_mean, weights_std
            best_mse_mean, best_mse_std = mse_mean, mse_std
            optimal_degree = i
        elif mse_mean < best_mse_mean:
            best_weights_mean, best_weights_std = weights_mean, weights_std
            best_mse_mean, best_mse_std = mse_mean, mse_std
            optimal_degree = i

    return optimal_degree, best_weights_mean, best_weights_std, best_mse_mean, best_mse_std