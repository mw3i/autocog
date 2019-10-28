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
    hidden_activation = (
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights']
            ),
            params['input']['hidden']['bias']
        )
    )

    hidden_activation2 = hps['hidden_activation'](
        np.add(
            np.add(
                np.matmul(
                    hidden_activation,
                    params['hidden']['inter']['weights']
                ),
                params['hidden']['inter']['bias']
            ),
            hidden_activation
        )
    )

    output_activation = hps['output_activation'](
        np.add(
            np.matmul(
                hidden_activation2,
                params['hidden']['output']['weights'],
            ),
            params['hidden']['output']['bias'],
        )
    )
    return [hidden_activation, hidden_activation2, output_activation]


## cost function (sum squared error)
def loss(params, inputs = None, targets = None, hps = None):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs, hps)[-1],
                targets
            )
        )
    ) / inputs.shape[0]



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
                'bias': np.random.uniform(*weight_range, [1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_categories]),
                'bias': np.random.uniform(*weight_range, [1, num_categories]),
            },
            'inter': {
                'weights': np.ones([num_hidden_nodes, num_hidden_nodes]),
                'bias': np.ones([1, num_hidden_nodes]),
            }
        },
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
    # import utils
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
        # 'A','A','A','A', 'B','B','B','B', # <-- type 1
        # 'A','A','B','B', 'B','B','A','A', # <-- type 2
        'A','A','A','B', 'B','B','B','A', # <-- type 4
        # 'B','A','A','B', 'A','B','B','A', # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    sigmoid = lambda x:  1 / (1 + np.exp(-x))
    sigmoid_deriv = lambda x:  sigmoid(x) * (1 - sigmoid(x))

    hps = {
        'lr': .5,  # <-- learning rate
        'weight_range': [-.3, .3],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': lambda x: 1 / (1 + np.exp(-x)),
        # 'hidden_activation2': lambda x: 1 / (1 + np.exp(-x)),

        'output_activation': lambda x: 1 / (1 + np.exp(-x)),
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        len(categories),
        weight_range = hps['weight_range']
    )


    num_epochs = 1000

    print('loss initially: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))

    
