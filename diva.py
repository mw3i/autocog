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

def forward(params, inputs = None, channels_indexed = None, hps = None):
    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights'],
            ),
            params['input']['hidden']['bias'],
        )
    )

    channel_activations = np.array([
        hps['output_activation'](
            np.add(
                np.matmul(
                    hidden_activation,
                    params['hidden']['output']['weights'][c,:,:],
                ),
                params['hidden']['output']['bias'][c,:,:],
            )
        ) 
    for c in channels_indexed
    ])

    return [hidden_activation, channel_activations]


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


def build_params(num_features, num_hidden_nodes, num_categories, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- number of category channels to make
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_categories, num_hidden_nodes, num_features]),
                'bias': np.zeros([num_categories, 1, num_features]),
            } 
        },
    }

def build_params_xavier(num_features, num_hidden_nodes, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- number of category channels to make
    '''
    return {
        'input': {
            'hidden': { # <-- xavier initialization for tanh activation function
                'weights': np.random.normal(0, 1, [num_features, num_hidden_nodes]) * np.sqrt(2 / (num_features + num_hidden_nodes)),
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.normal(0, 1, [num_categories, num_hidden_nodes, num_features]) * np.sqrt(2 / (num_hidden_nodes + num_features)),                
                'bias': np.zeros([num_categories, 1, num_features])
            }
        },
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params



def predict(params, inputs = None, targets = None, channels_indexed = None, hps = None):
    if np.any(targets) == None: targets = inputs
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

    # np.random.seed(0)

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
        # 'A','A','A','A', 'B','B','B','B', # <-- type 1
        'A','A','B','B', 'B','B','A','A', # <-- type 2
        # 'A','A','A','B', 'B','B','B','A', # <-- type 4
        # 'B','A','A','B', 'A','B','B','A', # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]

    hps = {
        'lr': .005,  # <-- learning rate
        'wr': [.01, .1],  # <-- weight range
        'num_hidden_nodes': 20,

        'hidden_activation': np.tanh,
        'output_activation': np.tanh,

        'beta': 1,
    }

    # params = build_params(
    #     inputs.shape[1],  # <-- num features
    #     hps['num_hidden_nodes'],
    #     len(categories),
    #     weight_range = hps['wr']
    # )

    params = build_params_xavier( # he initialization scheme
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        len(categories),
    )

    num_epochs = 10
    print('loss initially: ', loss(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = inputs, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

