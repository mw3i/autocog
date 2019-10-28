'''
DIVergent Autoencoder: GenDiscrim Version ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
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
                params['hidden']['output']['weights'],
            ),
            params['hidden']['output']['bias'],
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

    output_activation = 1 - hps['output_activation'](
        output_activation / output_activation.sum(axis=1, keepdims = True)
    )
    
    return [hidden_activation, channel_activations, output_activation]


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
        'num_hidden_nodes': 3,

        'hidden_activation': utils.sigmoid,
        'channel_activation': utils.linear,
        'output_activation': utils.softmax,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        len(categories),
        weight_range = hps['wr']
    )
   
    # params = build_params_xavier(
    #     inputs.shape[1],  # <-- num features
    #     hps['num_hidden_nodes'],
    #     len(categories),
    # )
   
    num_epochs = 100

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
