## std lib

## ext requirements
import autograd.numpy as np 



## - - - - - - - - - - - - - - - - - - - - - - -
# Convenience Functions
## - - - - - - - - - - - - - - -

def organize_data_from_txt(data_filepath, delimiter = ','):
    data = np.genfromtxt(data_filepath, delimiter = delimiter)

    data = {
        'inputs': data[:,:-1],
        'labels': data[:,-1],
        'categories': np.unique(data[:,-1]),
    }

    # map categories to label indices
    data['idx_map'] = {category: idx for category, idx in zip(data['categories'], range(len(data['categories'])))}

    # map original labels to label indices
    data['labels_indexed'] = [data['idx_map'][label] for label in data['labels']]

    # generate one hot targets
    data['one_hot_targets'] = np.eye(len(data['categories']))[data['labels_indexed']]

    return data


## - - - - - - - - - - - - - - - - - - - - - - -
# Activation FUNCTIONS
## - - - - - - - - - - - - - - -


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return x * (x > 0)

def linear(x):
    return x

def softmax(x):
    x -= np.max(x)
    return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T

def warp(x, num_dims):
    return np.exp(x - num_dims)

def softplus(x):
    return np.log(1 + np.exp(x))

