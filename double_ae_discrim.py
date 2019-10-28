'''
DIVergent Autoencoder ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
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

def forward(params, inputs = None, hps = None):
    hidden_activations = np.array([
        hps['hidden_activation'](
            np.add(
                np.matmul(
                    inputs,
                    params['input']['hidden']['weights'][c,:,:],
                ),
                params['input']['hidden']['bias'][c,:,:],
            )
        ) 
    for c in range(params['input']['hidden']['weights'].shape[0])
    ])

    channel_activations = np.array([
        hps['channel_activation'](
            np.add(
                np.matmul(
                    hidden_activations[c,:,:],
                    params['hidden']['output']['weights'][c,:,:],
                ),
                params['hidden']['output']['bias'][c,:,:],
            )
        ) 
    for c in range(params['input']['hidden']['weights'].shape[0])
    ])

    return [hidden_activations, channel_activations]


## sum squared error loss function
def loss(params, inputs = None, targets = None, channels = None, labels_indexed = None, hps = None):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs = inputs, hps = hps)[-1],
                targets
            )
        )
    )


## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_hidden_nodes, categories, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_features, num_hidden_nodes]),
                'bias': np.zero([len(categories), 1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_hidden_nodes, num_features]),
                'bias': np.zero([len(categories), 1, num_features]),
            }
        }
    }


def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params



def response(params, inputs = None, targets = None, channels = None, hps = None):
    if np.any(targets) == None: targets = inputs
    return np.argmin(
        np.sum(
            np.square(
                np.subtract(
                    targets,
                    forward(params, inputs = inputs, channels = channels, hps = hps)[-1]
                )
            ),
            axis = 2, keepdims = True
        ),
        axis = 0
    )[:,0]


def focus(params, inputs, channels, hps, beta = 0):
    outputs = forward(params, inputs = inputs, channels = channels, hps = hps)[-1]
    
    fweights = np.exp(beta * np.mean(outputs.std(axis=0), axis=0))
    fweights /= np.sum(fweights)

    return fweights


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils

    inputs = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0],

        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
    ])

    labels = [
        'A','A','A','A', 'B','B','B','B', # <-- type 1
        # 'A','A','B','B', 'B','B','A','A', # <-- type 2
        # 'A','A','A','B', 'B','B','B','A', # <-- type 4
        # 'B','A','A','B', 'A','B','B','A', # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [.01, .1],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': utils.sigmoid,
        'channel_activation': utils.linear,

        'beta': 1,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        categories,
        weight_range = hps['wr']
    )
    
    num_epochs = 100


    print('loss initially: ', loss(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps))

