'''
WARP | Kurtz & Silliman
3 Critical Functions:
    forward(...) <-- generates output from input data
    loss(...) <-- calculates success at predicting class labels
    loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    build(...) <-- generates a dictionary of random parameters (connections)
    update_params(...) <-- updates param weights based on gradients provided
'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad

## produces model outputs
def forward(params, inputs = None, hps = None):
    hidden_activation = warp(
        np.matmul(
            inputs,
            params['input']['hidden']['weights']
        ),
        inputs.shape[1]
    )

    output_activation = hps['classifier_activation'](
        np.matmul(
            hidden_activation,
            params['hidden']['output']['weights'],
        )
    )
    return [hidden_activation, output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    ## Cross-Entropy (i think); usually explodes
    # o = forward(params, inputs = inputs, hps = hps)[-1]
    # c = o * targets + (1 - o) * (1 - targets)
    # return -np.sum(np.log(c))
    ## SSE
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs=inputs, hps = hps)[-1],
                targets
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
    num_categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_categories]),
            } 
        },
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
    return params


def response(params, inputs = None, hps = None):
    return np.argmax(
        forward(params, inputs = inputs, hps = hps)[-1],
        axis = 1
    )


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    def warp(x, num_dims):
        return np.exp(-(x - num_dims))

    def softmax(x):
        x -= np.max(x)
        return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T
    
    inputs = np.array([
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.9],
        [0.1, 0.9, 0.1],
        [0.9, 0.1, 0.1],

        [0.9, 0.9, 0.9],
        [0.9, 0.9, 0.1],
        [0.9, 0.1, 0.9],
        [0.1, 0.9, 0.9],
    ])

    labels = np.array([
        [0,1],
        [0,1],
        [0,1],
        [0,1],

        [1,0],
        [1,0],
        [1,0],
        [1,0],
    ])
    idx_labels = np.argmax(labels,axis=1)

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-.1, .1], # <-- weight range
        'num_hidden_nodes': 10,

        'classifier_activation': softmax,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        labels.shape[1]
    )

    num_epochs = 1000

    print('loss initially: ', loss(params, inputs = inputs, targets = labels, hps = hps))
    
    import matplotlib.pyplot as plt 

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = labels, hps = hps)
        params = update_params(params, gradients, hps['lr'])
        
    print('loss after training: ', loss(params, inputs = inputs, targets = labels, hps = hps))

    print(
        'predictions:',
        np.argmax(forward(params, inputs = inputs, hps = hps)[-1], axis=1)
    )
    print(
        'labels:',
        idx_labels
    )
    
    print(
        'accuracy:',
        np.mean(
            np.equal(
                np.argmax(forward(params, inputs = inputs, hps = hps)[-1], axis=1),
                idx_labels
            )
        )
    )
