'''
Basic Multilayer Classifier (ie, multilayer logistic regression)
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
    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights']
            ),
            params['input']['hidden']['bias']
        )
    )

    output_activation = hps['classifier_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['output']['weights'],
            ),
            params['hidden']['output']['bias'],
        )
    )
    return [hidden_activation, output_activation]


## cross entropy loss function
def loss(params, inputs = None, targets = None, hps = None):
    model_output = forward(params, inputs = inputs, hps = hps)[-1]
    return -np.sum(
        targets * np.log(model_output),
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
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_categories]),
                'bias': np.zeros([1, num_categories]),
            } 
        },
    }

def build_params_xavier(num_features, num_hidden_nodes, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.random.normal(0, 1, [num_features, num_hidden_nodes]) * np.sqrt(2 / (num_features + num_hidden_nodes)),
                'bias': np.zeros([1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.normal(0, 1, [num_hidden_nodes, num_categories]) * np.sqrt(2 / (num_hidden_nodes + num_categories)),
                'bias': np.zeros([1, num_categories]),
            } 
        },
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params

def update_params_swarm(params, gradients, lr_unit, lr_population):
    for layer in params:
        for connection in params[layer]:

            # gradient descent + particle swarm optimization
            params[layer][connection]['weights'] -= (lr_unit * gradients[layer][connection]['weights']) + (lr_population * np.mean(gradients[layer][connection]['weights'], axis = -1, keepdims = True))

            # bias weights update normally
            params[layer][connection]['bias'] -= lr_unit * gradients[layer][connection]['bias']
    return params

def response(params, inputs = None, hps = None):
    return np.argmax(
        forward(params, inputs = inputs, hps = hps)[-1],
        axis = 1
    )


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils
    np.random.seed(5)
    
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
        'lr': .05,  # <-- learning rate
        'wr': [-.1, .1], # <-- weight range
        'num_hidden_nodes': 5,

        'hidden_activation': np.tanh,
        'classifier_activation': utils.softmax,
    }

    # params = build_params(
    #     inputs.shape[1],  # <-- num features
    #     hps['num_hidden_nodes'],
    #     labels.shape[1]
    # )

    params = build_params_xavier(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        labels.shape[1]
    )

    num_epochs = 100

    print('loss initially: ', loss(params, inputs = inputs, targets = labels, hps = hps))
    
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = labels, hps = hps)
        # params = update_params(params, gradients, hps['lr'])
        params = update_params_swarm(params, gradients, hps['lr'], 3)
    
    print('loss after training: ', loss(params, inputs = inputs, targets = labels, hps = hps))

    
