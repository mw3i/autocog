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

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

def forward(params, inputs = None, hps = None):
    hidden_activation = np.array([
        np.exp(
            -np.matmul(
                np.subtract(
                    inputs,
                    params['input']['hidden']['bias'][:,h]
                ) ** 2,
                params['input']['hidden']['weights'][:,h],
            )
        ) for h in range(params['input']['hidden']['weights'].shape[1])
    ]).T


    channel_activations = hps['channel_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['categories']['weights'],
            ),
            params['hidden']['categories']['bias'],
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


## logistic loss function
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


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':

    inputs = np.array([
        np.concatenate([np.random.normal(.6,.1,[50]), np.random.normal(.4,.1,[50])]),
        np.concatenate([np.random.normal(.3,.1,[50]), np.random.normal(.9,.1,[50])]),
    ]).T

    labels = ['a'] * 50 + ['b'] * 50

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .001,  # <-- learning rate
        'wr': [.01, .3],  # <-- weight range
        'num_hidden_nodes': 4,

        'hidden_activation': lambda x: x,
        'channel_activation': lambda x: x,
        'output_activation': softmax,
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        categories,
        weight_range = hps['wr']
    )
   

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

    n = np.linspace(-2,2,100)
    import matplotlib.pyplot as plt 

    f = forward(params, inputs = np.array([n,n]).T, hps = hps)[0]
    for h in range(hps['num_hidden_nodes']):
        plt.plot(
            n,
            f[:,h],
        )
    # plt.show()
