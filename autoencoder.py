'''
Basic Autoencoder
3 Critical Functions:
    forward(...) <-- generates output from input data
    loss(...) <-- calculates success at reconstructing the input data
    loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    build(...) <-- generates a dictionary of random parameters (connections)
    utils.update_params(...) <-- updates param weights based on gradients provided
'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad


## produces model outputs
def forward(params, inputs = None, hps = None):

    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['encode']['decode']['weights'],
            ),
            params['encode']['decode']['bias'],
        )
    )
    
    output_activation = hps['output_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['decode']['output']['weights'],
            ),
            params['decode']['output']['bias'],
        )
    )
    return [hidden_activation, output_activation]


## sum squared error loss function
def loss(params, inputs = None, targets = None, hps = None):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs = inputs, hps = hps)[-1],
                targets,
            )
        )
    )


## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_hidden_nodes, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'encode': {
            'decode': {
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'decode': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_features]),
                'bias': np.zeros([1, num_features]),
            } 
        }
    }

def build_params_xavier(num_features, num_hidden_nodes):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'encode': {
            'decode': {
                'weights': np.random.normal(0, 1, [num_features, num_hidden_nodes]) * np.sqrt(2 / (num_features + num_hidden_nodes)),
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'decode': {
            'output': {
                'weights': np.random.normal(0, 1, [num_hidden_nodes, num_features]) * np.sqrt(2 / (num_hidden_nodes + num_features)),
                'bias': np.zeros([1, num_features]),
            } 
        }
    }

# - - - - - - - - - - - - - - - - - -

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params

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

    hps = {
        'lr': .005,  # <-- learning rate
        'wr': [.01, .1],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': np.tanh,
        'output_activation': utils.linear,
    }

    # params = build_params(
    #     inputs.shape[1],  # <-- num features
    #     hps['num_hidden_nodes'],
    #     weight_range = hps['wr']
    # )

    params = build_params_xavier(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
    )
    
    num_epochs = 100

    print('loss initially: ', loss(params, inputs = inputs, targets = inputs, hps = hps))
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = inputs, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    print('loss after training: ', loss(params, inputs = inputs, targets = inputs, hps = hps))

    
