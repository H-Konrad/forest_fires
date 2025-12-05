import numpy as np

def ridge_regression(inputs, outputs, alpha):
    return np.linalg.solve((inputs.T@inputs + alpha * np.identity(inputs.shape[1])), inputs.T@outputs)

def pre_poly(training_data, testing_data, degree):
    training_data_dict, testing_data_dict = {}, {}
    for i in range(1, degree + 1):
        training_data_list, testing_data_list = [], []
        for j in range(len(testing_data)):
            training_data_list.append((poly_data(training_data[j][0], i), training_data[j][1]))
            testing_data_list.append((poly_data(testing_data[j][0], i), testing_data[j][1]))

        training_data_dict[i] = training_data_list
        testing_data_dict[i] = testing_data_list

    return training_data_dict, testing_data_dict

def ridge_analysis(training_data, testing_data, alpha_range = np.linspace(0, 100, 1001)):
    for key, value in training_data.items():
        for i in alpha_range:
            weights_list, mse_list = [], []
            for j in range(len(value)):
                weights = ridge_regression(value[j][0], value[j][1], i)
                weights_list.append(weights)
                mse = prediction_error(testing_data[key][j][0], testing_data[key][j][1], weights)
                mse_list.append(mse)

            weights_mean = np.mean(weights_list, axis = 0)
            weights_std = np.std(weights_list, axis = 0)
            mse_mean = np.mean(mse_list, axis = 0)
            mse_std = np.std(mse_list, axis = 0)

            if key == 1:
                best_weights_mean, best_weights_std = weights_mean, weights_std
                best_mse_mean, best_mse_std = mse_mean, mse_std
                optimal_degree, optimal_alpha = key, i
            elif mse_mean < best_mse_mean:
                best_weights_mean, best_weights_std = weights_mean, weights_std
                best_mse_mean, best_mse_std = mse_mean, mse_std
                optimal_degree, optimal_alpha = key, i

    return optimal_degree, optimal_alpha, best_weights_mean, best_weights_std, best_mse_mean, best_mse_std