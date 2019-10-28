'''
DIVergent Autoencoder ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
    ** SINGLE LAYER EDITION **

3 Critical Functions:
    forward(...) <-- generates DIVA's output from input data
    loss(...) <-- calculates DIVA's success at reconstructing the input data
    loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    build(...) <-- generates a dictionary of random parameters (connections)
    update_params(...) <-- updates param weights based on gradients provided
    response(...) <-- get diva's response probabilities (ie, classifications)
'''
## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad


def forward(params, inputs = None, channels_indexed = None, hps = None):
    channel_activations = np.array([
        hps['channel_activation'](
            np.add(
                np.matmul(
                    inputs,
                    params['input']['output']['weights'][c,:,:],
                ),
                params['input']['output']['bias'][c,:,:],
            )
        ) 
    for c in channels_indexed
    ])

    return [channel_activations]


## sum squared error loss function
def loss(params, inputs = None, targets = None, channels_indexed = None, labels_indexed = None, hps = None):
    if labels_indexed == None:
        labels_indexed = np.zeros([1,inputs.shape[0]], dtype=int)

    channel_activations = forward(params, inputs = inputs, channels_indexed = channels_indexed, hps = hps)[-1]
    channel_activation = channel_activations[labels_indexed, range(inputs.shape[0]),:]

    return np.sum(
        np.square(
            np.subtract(
                channel_activation,
                targets,
            )
        )
    )



## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'output': {
                'weights': np.zeros([len(categories), num_features, num_features]),
                'bias': np.zeros([len(categories), 1, num_features]),
            }
        }
    }

def build_params_xavier(num_features, categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'output': {
                'weights': np.random.normal(0, 1, [len(categories), num_features, num_features]) * np.sqrt(1 / (num_features)),
                'bias': np.zeros([len(categories), 1, num_features]),
            }
        }
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            gradients[layer][connection]['weights'] = gradients[layer][connection]['weights'] * (1 - np.eye(gradients[layer][connection]['weights'].shape[1]))
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


def predict(params, inputs = None, targets = None, channels_indexed = None, hps = None):
    if np.any(targets) == None: 
        targets = inputs
    return np.argmin(
        np.sum(
            np.square(
                np.subtract(
                    targets,
                    forward(params, inputs = inputs, channels_indexed = channels_indexed, hps = hps)[-1]
                )
            ),
            axis = 2, keepdims = True
        ),
        axis = 0
    )[:,0]







# - - - - - - - - - - - - - - - - - -






if __name__ == '__main__':
    import utils

    # data = np.genfromtxt('iris.csv', delimiter = ',')
    data = np.array([
        [0,0,0,1],
        [0,0,1,1],
        [0,1,0,1],
        [0,1,1,1],
        [1,1,1,2],
        [1,1,0,2],
        [1,0,1,2],
        [1,0,0,2],
    ])
    inputs = data[:,:-1]
    labels = data[:,-1]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-.1, .1],  # <-- weight range
        'channel_activation': utils.linear,
    }

    params = build_params_xavier(
        inputs.shape[1],  # <-- num features
        categories,
    )
    
    num_epochs = 100

    print('loss initially: ', loss(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

    acc = []
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps)
        params = update_params(params, gradients, hps['lr'])

        acc.append(
            np.mean(
                np.equal(
                    predict(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), hps = hps),
                    labels_indexed,
                )
            )
        )

    print('loss after training: ', loss(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

    # print(np.round(params['input'][1]['weights'], 3))

    print(
        'response:\n', 
        predict(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), hps = hps)
    )
