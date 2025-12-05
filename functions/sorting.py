import numpy as np

def standardise(data):
    std = np.std(data, axis = 0)
    mean = np.mean(data, axis = 0)
    return (data - mean)/(std), mean, std

def unstandardise(data, mean, std):
    return (data * std) + mean

def k_fold_splits(data, k):
    indexes = np.random.permutation(len(data))
    m, r = divmod(len(data), k)
    splits = [indexes[i * m + min(i, r):(i + 1) * m + min(i + 1, r)] for i in range(k)]

    columns = data.shape[1]
    training_data, testing_data = [], []
    for i in range(k):
        training_split = np.concatenate([splits[j] for j in range(k) if j != i])
        training_data.append((data[training_split][:, 0: columns-1], data[training_split][:, columns-1]))
        testing_data.append((data[splits[i]][:, 0: columns-1], data[splits[i]][:, columns-1]))

    return training_data, testing_data
