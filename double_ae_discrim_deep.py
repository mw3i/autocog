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
    hidden1_activations = np.array([
        hps['hidden1_activation'](
            np.add(
                np.matmul(
                    inputs,
                    params['input']['hidden1']['weights'][c,:,:],
                ),
                params['input']['hidden1']['bias'][c,:,:],
            )
        ) 
    for c in range(params['input']['hidden1']['weights'].shape[0])
    ])

    hidden2_activations = np.array([
        hps['hidden2_activation'](
            np.add(
                np.matmul(
                    hidden1_activations[c,:,:],
                    params['hidden1']['hidden2']['weights'][c,:,:],
                ),
                params['hidden1']['hidden2']['bias'][c,:,:],
            )
        ) 
    for c in range(params['hidden1']['hidden2']['weights'].shape[0])
    ])

    channel_activations = np.array([
        hps['channel_activation'](
            np.add(
                np.matmul(
                    hidden2_activations[c,:,:],
                    params['hidden2']['output']['weights'][c,:,:],
                ),
                params['hidden2']['output']['bias'][c,:,:],
            )
        ) 
    for c in range(params['hidden2']['output']['weights'].shape[0])
    ])

    ## reconstructive error
    output_activation = np.sum(
        np.square(
            np.subtract(
                inputs,
                channel_activations,
            )
        ),
        axis = 2
    ).T

    output_activation = 1 - hps['classifier_activation'](
        output_activation / output_activation.sum(axis=1, keepdims = True)
    )

    return [hidden1_activations, hidden2_activations, channel_activations, output_activation]


## sum squared error loss function
def loss(params, inputs = None, targets = None, hps = None):
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


def build_params(num_features, num_hidden1_nodes, num_hidden2_nodes, categories, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden1': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_features, num_hidden1_nodes]),
                'bias': np.zeros([len(categories), 1, num_hidden1_nodes]),
            },
        },
        'hidden1': {
            'hidden2': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_hidden1_nodes, num_hidden2_nodes]),
                'bias': np.zeros([len(categories), 1, num_hidden2_nodes]),
            },
        },
        'hidden2': {
            'output': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_hidden2_nodes, num_features]),
                'bias': np.zeros([len(categories), 1, num_features]),
            }
        }
    }

def build_params_xavier(num_features, num_hidden1_nodes, num_hidden2_nodes, categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden1': {
                'weights': np.random.normal(0, 1, [len(categories), num_features, num_hidden1_nodes]) * np.sqrt(2 / (num_features + num_hidden1_nodes)),
                'bias': np.zeros([len(categories), 1, num_hidden1_nodes]),
            },
        },
        'hidden1': {
            'hidden2': {
                'weights': np.random.normal(0, 1, [len(categories), num_hidden1_nodes, num_hidden2_nodes]) * np.sqrt(2 / (num_hidden1_nodes + num_hidden2_nodes)),
                'bias': np.zeros([len(categories), 1, num_hidden2_nodes]),
            },
        },
        'hidden2': {
            'output': {
                'weights': np.random.normal(0, 1, [len(categories), num_hidden2_nodes, num_features]) * np.sqrt(2 / (num_hidden2_nodes + num_features)),
                'bias': np.zeros([len(categories), 1, num_features]),
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


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils

    np.random.seed(0)

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
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .5,  # <-- learning rate
        'wr': [-.1, .1],  # <-- weight range
        'num_hidden1_nodes': 4,
        'num_hidden2_nodes': 4,

        'hidden1_activation': utils.sigmoid,
        'hidden2_activation': utils.sigmoid,
        'channel_activation': utils.linear,
        'classifier_activation': utils.softmax,

        'beta': 1,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden1_nodes'],
        hps['num_hidden2_nodes'],
        categories,
        weight_range = hps['wr']
    )
    
    # params = build_params_xavier(
    #     inputs.shape[1],  # <-- num features
    #     hps['num_hidden1_nodes'],
    #     hps['num_hidden2_nodes'],
    #     categories,
    # )

    num_epochs = 100


    print('loss initially: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets,  hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))

