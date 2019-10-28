'''
DIVergent Autoencoder ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
    *** DETANGLER EDITION ***

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


## produces model outputs
def forward(params, inputs = None, channels = None, hps = None):

    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights'],
            ),
            params['input']['hidden']['bias'],
        )
    )

    hidden2_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['hidden2']['weights'],
            ),
            params['hidden']['hidden2']['bias'],
        )
    )

    channel_activations = np.array([
        hps['channel_activation'](
            np.add(
                np.add(
                    np.matmul(
                        hidden_activation,
                        params['hidden'][channel]['weights'],
                    ),
                    params['hidden'][channel]['bias'],
                ),    
                np.add(
                    np.matmul(
                        hidden2_activation,
                        params['hidden2'][channel]['weights'],
                    ),
                    params['hidden2'][channel]['bias'],
                )
            )
        )
    for channel in channels
    ])

    return [hidden_activation, hidden2_activation, channel_activations]


## sum squared error loss function
def loss(params, inputs = None, targets = None, channels = None, labels_indexed = None, hps = None):
    if labels_indexed == None:
        labels_indexed = np.zeros([1,inputs.shape[0]], dtype=int)

    channel_activations = forward(params, inputs = inputs, channels = channels, hps = hps)[-1]
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


def build_params(num_features, num_hidden_nodes, num_hidden2_nodes, categories, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.random.uniform(*weight_range, [1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'hidden2': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_hidden2_nodes]),
                'bias': np.random.uniform(*weight_range, [1, num_hidden2_nodes]),
            },
            **{
                category: { # <-- direct connections
                    'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_features]),
                    'bias': np.random.uniform(*weight_range, [1, num_features]),
                }
                for category in categories
            }
        },
        'hidden2': {
            category: { # <-- detangled connections
                'weights': np.random.uniform(*weight_range, [num_hidden2_nodes, num_features]),
                'bias': np.random.uniform(*weight_range, [1, num_features]),
            }
            for category in categories
        },
    }


def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params



def response(params, inputs = None, targets = None, channels = None, hps = None):
    if np.any(targets) == False: targets = inputs
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


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils 

    inputs = np.array([
        [.2, .3],
        [.3, .4],
        [.4, .5],
        [.5, .6],
        [.6, .7],
        [.7, .8],
        [.8, .9],

        [.2, .1],
        [.3, .2],
        [.4, .3],
        [.5, .4],
        [.6, .5],
        [.7, .6],
        [.8, .7],
    ])


    labels = [
        'A','A','A','A','A','A','A',   'B','B','B','B','B','B','B',
    ]
    categories = list(set(labels))
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]


    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-.1, .1],  # <-- weight range
        'num_hidden_nodes': 4,
        'num_hidden2_nodes': 10,

        'hidden_activation': utils.sigmoid,
        'channel_activation': utils.linear,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        hps['num_hidden2_nodes'],
        categories,
        weight_range = hps['wr']
    )
    
    num_epochs = 100

    print('loss initially: ', loss(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = inputs, channels = categories, labels_indexed = labels_indexed, hps = hps))
