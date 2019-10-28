'''
DIVergent Autoencoder ([Kurtz 2017](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf))
    *** DIVA THOERY OF TASK EDITION ***

3 Critical Functions:
    forward(...) <-- generates DIVA's output from input data
    loss(...) <-- calculates DIVA's success at reconstructing the input data
    loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    build(...) <-- generates a dictionary of random parameters (connections)
    update_params(...) <-- updates param weights based on gradients provided
'''
## ext requirements
import autograd.numpy as np 
from autograd import grad


def forward(params, inputs = None, channels_indexed = None, hps = None):
    hidden_activation = hps['hidden_activation'](
        np.add(
            np.matmul(
                inputs,
                params['input']['hidden']['weights'],
            ),
            params['input']['hidden']['bias'],
        )
    )

    channel_activations = np.array([
        hps['channel_activation'](
            np.add(
                np.matmul(
                    hidden_activation,
                    params['hidden']['channels']['weights'][c,:,:],
                ),
                params['hidden']['channels']['bias'][c,:,:],
            )
        ) 
        for c in channels_indexed
    ])

    classifier_activation = hps['classifier_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['classifier']['weights'],
            ),
            params['hidden']['classifier']['bias'],
        )
    )

    return [hidden_activation, channel_activations, classifier_activation]



## sum squared error loss function
def loss(params, inputs = None, inference_targets = None, classification_targets = None, channels_indexed = None, labels_indexed = None, hps = None):
    if labels_indexed == None:
        labels_indexed = np.zeros([1,inputs.shape[0]], dtype=int)
    
    hidden_activation, channel_activations, classifier_activation = forward(params, inputs = inputs, channels_indexed = channels_indexed, hps = hps)
    channel_activation = channel_activations[labels_indexed, range(inputs.shape[0]),:]

    return np.add(
        (1 - hps['dratio']) * np.sum(
            np.square(
                np.subtract(
                    channel_activation,
                    inference_targets,
                )
            )
        ),
        hps['dratio'] * np.sum(
            np.square(
                np.subtract(
                    classifier_activation,
                    classification_targets
                )
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
    categories <-- (list) list of category labels to use as keys for decode -- output connections
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
            'channels': {
                'weights': np.random.uniform(*weight_range, [num_categories, num_hidden_nodes, num_features]),
                'bias': np.zeros([num_categories, 1, num_features]),
            },
            'classifier': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_categories]),
                'bias': np.zeros([1, num_categories]),
            },
        }
    }

def build_params_xavier(num_features, num_hidden_nodes, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    categories <-- (list) list of category labels to use as keys for decode -- output connections
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
            'channels': {
                'weights': np.random.normal(0, 1, [num_categories, num_hidden_nodes, num_features]) * np.sqrt(2 / (num_categories + num_hidden_nodes)),
                'bias': np.zeros([num_categories, 1, num_features]),
            },
            'classifier': {
                'weights': np.random.normal(0, 1, [num_hidden_nodes, num_categories]) * np.sqrt(2 / (num_hidden_nodes + num_categories)),
                'bias': np.zeros([1, num_categories]),
            },
        }
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


def predict(params, inputs = None, channels_indexed = None, hps = None):
    return np.argmax(
        forward(params, inputs = inputs, channels_indexed = channels_indexed, hps = hps)[-1],
        axis = 1
    )


def diva_response(params, inputs, targets = None, channels_indexed = None, hps = None):
    if targets == None: targets = inputs
    return np.argmin(
        np.sum(
            np.square(
                np.subtract(
                    targets,
                    forward(params, inputs = inputs, channels_indexed = channels_indexed, hps = hps)[-2]
                )
            ),
            axis = 2, keepdims = True
        ),
        axis = 0
    )[:,0]



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

    labels = [
        'A','A','A','A','A','A','A',   'B','B','B','B','B','B','B',
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .1,  # <-- learning rate
        'wr': [-2, 2],  # <-- weight range
        'num_hidden_nodes': 2,

        'hidden_activation': np.tanh,
        'channel_activation': np.tanh,
        'classifier_activation': utils.softmax,

        'dratio': .5,
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

    print('loss initially: ', loss(params, inputs = inputs, inference_targets = inputs, classification_targets = one_hot_targets, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, inference_targets = inputs, classification_targets = one_hot_targets, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps)
        params = update_params(params, gradients, hps['lr'])

    print('loss after training: ', loss(params, inputs = inputs, inference_targets = inputs, classification_targets = one_hot_targets, channels_indexed = list(idx_map.values()), labels_indexed = labels_indexed, hps = hps))

    # hidden_activation, channel_activations, classifier_activation = forward(params, inputs = inputs, channels_indexed = list(idx_map.values()), hps = hps)
    # print('classifier activation:\n', np.round(classifier_activation, 0))




