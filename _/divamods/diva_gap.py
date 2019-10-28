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
    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights'],
            ),
            params['input']['hidden']['bias'],
        )
    )

    channel_activations = hps['channel_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['categories']['weights'],
            ),
            params['hidden']['categories']['bias'],
        )
    ) 

    ## global average pooling
    output_activation = hps['output_activation'](
        np.mean(
            channel_activations,
            axis = 2,
        ).T
    )
    return [hidden_activation, channel_activations, output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    o = forward(params, inputs = inputs, hps = hps)[-1]
    c = o * targets + (1 - o) * (1 - targets)
    return -np.sum(np.log(c))

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
                'weights': np.random.uniform(*weight_range, [num_features, num_hidden_nodes]),
                'bias': np.random.uniform(*weight_range, [1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'categories': {
                'weights': np.random.uniform(*weight_range, [len(categories), num_hidden_nodes, num_features]),
                'bias': np.random.uniform(*weight_range, [len(categories), 1, num_features]),
            } 
        },
    }


def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params



# def response(params, inputs = None, targets = None, channels = None, hps = None):
#     if np.any(targets) == None: targets = inputs
#     return np.argmin(
#         np.sum(
#             np.square(
#                 np.subtract(
#                     targets,
#                     forward(params, inputs = inputs, channels = channels, hps = hps)[-1]
#                 )
#             ),
#             axis = 2, keepdims = True
#         ),
#         axis = 0
#     )[:,0]







# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils

    # data = np.genfromtxt('iris.csv', delimiter = ',')
    # data = np.genfromtxt('mamm.csv', delimiter = ',')
    data = np.genfromtxt('leaf.csv', delimiter = ',')


    inputs = data[:,:-1]
    labels = data[:,-1]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-.1, .1],  # <-- weight range
        'num_hidden_nodes': 20,

        'hidden_activation': np.tanh,
        'channel_activation': utils.relu,
        'output_activation': utils.softmax,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        categories,
        weight_range = hps['wr']
    )
    
    num_epochs = 4000

    print('loss initially: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    print(
        np.mean(
            np.equal(
                np.argmax(
                    forward(params, inputs = inputs, hps = hps)[-1],
                    axis = 1
                ),
                labels_indexed
            )
        )
    )

