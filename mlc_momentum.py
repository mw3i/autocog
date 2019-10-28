'''
Basic Multilayer Classifier w/ Multiple Param Update Schemes
    * normal gradient descent
    * gradient descent w/ momentum
        ^ based of this wikipedia page: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
        ^ and this tutorial: https://engmrk.com/gradient-descent-with-momentum/ <-- by Muhammad Rizwan
        ^ and: https://towardsdatascience.com/a-bit-beyond-gradient-descent-mini-batch-momentum-and-some-dude-named-yuri-nesterov-a3640f9e496b <-- by Joseph J. Bautista
    * gradient descent w/ momentum & particle swarm optimization at the hidden layer
        ^ based off the wikipedia page for PSO and general intuition about how that might apply to nn gradients

'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad


## produces model outputs
def forward(params, inputs = None, hps = None):
    hidden_activation = hps['hidden_activation'](
        np.add(            np.matmul(
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


## logistic loss function
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
    for layer in gradients:
        for connection in gradients[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


def update_params_swarm(params, gradients, lr_unit, lr_swarm):
    #__hidden layer (hidden weights get PSO update)
    params['input']['hidden']['weights'] -= (lr_unit * gradients['input']['hidden']['weights']) + (lr_swarm * np.mean(gradients['input']['hidden']['weights'], axis = -1, keepdims = True)) # <-- particle swarm optimization term
    params['input']['hidden']['bias'] -= lr_unit * gradients['input']['hidden']['bias']

    #__output layer (update normally)
    params['hidden']['output']['weights'] -= lr_unit * gradients['hidden']['output']['weights']
    params['hidden']['output']['bias'] -= lr_unit * gradients['hidden']['output']['bias']

    return params


# ##__Wikipedia's Description
# def update_params_momentum(params, gradients, velocities, lr_unit, lr_intertia):
#     for layer in gradients:
#         for connection in gradients[layer]:
            
#             velocities[layer][connection]['weights'] = (lr_unit * gradients[layer][connection]['weights']) + (lr_intertia * velocities[layer][connection]['weights'])
#             velocities[layer][connection]['bias'] = (lr_unit * gradients[layer][connection]['bias']) + (lr_intertia * velocities[layer][connection]['bias'])
            
#             params[layer][connection]['weights'] -= velocities[layer][connection]['weights']
#             params[layer][connection]['bias'] -= velocities[layer][connection]['bias']

#     return params, velocities


# def update_params_momentum_pso(params, gradients, velocities, lr_unit, lr_intertia, lr_swarm):
#     for layer in gradients:
#         for connection in gradients[layer]:
            
#             velocities[layer][connection]['weights'] = (lr_unit * gradients[layer][connection]['weights']) + (lr_intertia * velocities[layer][connection]['weights'])
#             velocities[layer][connection]['bias'] = (lr_unit * gradients[layer][connection]['bias']) + (lr_intertia * velocities[layer][connection]['bias'])
            
#             params[layer][connection]['weights'] -= velocities[layer][connection]['weights']
#             params[layer][connection]['bias'] -= velocities[layer][connection]['bias']


#     params['input']['hidden']['weights'] -= lr_swarm * np.mean(velocities['input']['hidden']['weights'], axis = -1, keepdims = True) # <-- particle swarm optimization term
#     return params, velocities


##__Muhammad Rizwan's Implementation
def update_params_momentum(params, gradients, velocities, lr_unit, beta):
    for layer in gradients:
        for connection in gradients[layer]:
            
            velocities[layer][connection]['weights'] = ((1-beta) * gradients[layer][connection]['weights']) + (beta * velocities[layer][connection]['weights'])
            velocities[layer][connection]['bias'] = ((1-beta) * gradients[layer][connection]['bias']) + (beta * velocities[layer][connection]['bias'])
            
            params[layer][connection]['weights'] -= lr_unit * velocities[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr_unit * velocities[layer][connection]['bias']

    return params, velocities


def update_params_momentum_pso(params, gradients, velocities, lr_unit, beta, lr_swarm):
    for layer in gradients:
        for connection in gradients[layer]:
            
            velocities[layer][connection]['weights'] = ((1-beta) * gradients[layer][connection]['weights']) + (beta * velocities[layer][connection]['weights'])
            velocities[layer][connection]['bias'] = ((1-beta) * gradients[layer][connection]['bias']) + (beta * velocities[layer][connection]['bias'])
            
            params[layer][connection]['weights'] -= lr_unit * velocities[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr_unit * velocities[layer][connection]['bias']


    params['input']['hidden']['weights'] -= lr_swarm * np.mean(velocities['input']['hidden']['weights'], axis = -1, keepdims = True) # <-- particle swarm optimization term
    return params, velocities


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

    
