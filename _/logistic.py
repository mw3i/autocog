'''
Basic Logistic Regression
3 Critical Functions:
    logistic.forward(...) <-- generates output from input data
    logistic..loss(...) <-- calculates success at predicting class labels
    logistic..loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    logistic.build(...) <-- generates a dictionary of random parameters (connections)
    utils.update_params(...) <-- updates param weights based on gradients provided
'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad


## produces model outputs
def forward(params, inputs = None, hps = None):
    output_activation = hps['output_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['output']['weights'],
            ),
            params['input']['output']['bias'],
        )
    )
    return [output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    o = forward(params, inputs = inputs, hps = hps)[-1]
    c = o * targets + (1 - o) * (1 - targets)
    return -np.sum(np.log(c))


## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_classes):
    '''
    num_features <-- (numeric) number of feature in the dataset
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'output': {
                'weights': np.zeros([num_features, num_classes]),
                'bias': np.zeros([1, num_classes]),
            }
        }
    }


def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


def response(params, inputs = None, hps = None):
    return np.argmax(
        forward(params, inputs = inputs, hps = hps)[-1],
        axis = 1
    )


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

    labels = np.array([
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],
        [0,1],

        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
        [1,0],
    ])


    hps = {
        'lr': .5,  # <-- learning rate
        'output_activation': utils.sigmoid,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        labels.shape[1]
    )

    num_epochs = 1000

    print('loss initially: ', loss(params, inputs = inputs, targets = labels, hps = hps))
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = labels, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    print('loss after training: ', loss(params, inputs = inputs, targets = labels, hps = hps))

    
