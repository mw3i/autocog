'''
DIVergent Autoencoder: Shallow GenDiscrim Version ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
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
    channel_activations = hps['channel_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['categories']['weights'],
            ),
            params['input']['categories']['bias'],
        )
    ) 

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

    output_activation = hps['output_activation'](
        1 - (output_activation / output_activation.sum(axis=1, keepdims = True))
    )
    return [channel_activations, output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    o = forward(params, inputs = inputs, hps = hps)[-1]
    c = o * targets + (1 - o) * (1 - targets)
    return -np.sum(np.log(c))

## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, categories, weight_range = [-.1, .1]):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'categories': {
                'weights': np.zeros([len(categories), num_features, num_features]),
                'bias': np.zeros([len(categories), 1, num_features]),
            } 
        },
    }


def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params



# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils

    data = data = np.array([
        [0,0,0,1],
        [0,0,1,1],
        [0,1,0,1],
        [1,0,0,1],
        [1,1,1,2],
        [1,1,0,2],
        [1,0,1,2],
        [0,1,1,2],
    ])

    inputs = data[:,:-1]
    labels = data[:,-1]

    categories = np.unique(data[:,-1])
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [-3, 3],  # <-- weight range

        'channel_activation': utils.linear,
        'output_activation': utils.softmax,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        categories,
        weight_range = hps['wr']
    )
   

    num_epochs = 1000

    print('loss initially: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('loss after training: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    print('model predictions:\n', 
        *[categories[l] for l in np.argmax(
            forward(params, inputs = inputs, hps = hps)[-1],
            axis = 1,
        )]
    )

