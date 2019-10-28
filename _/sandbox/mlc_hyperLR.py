'''
MLC toying around with hypergradient descent

main things that are different from normal version:
    * learning rate is a dictionary of values for each layer
    * addition of a "hyper learning rate" parameter
    * lr update function

problems:
    * hyper learning rate has to be set very very low in order for this to do anything
        ^ and at that point... is it even doing anything?

'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad
import copy

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

    output_activation = hps['output_activation'](
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
                'bias': np.random.uniform(*weight_range, [1, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.uniform(*weight_range, [num_hidden_nodes, num_categories]),
                'bias': np.random.uniform(*weight_range, [1, num_categories]),
            } 
        },
    }

def update_params(params, gradients, lrs):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lrs[layer][connection]['weights'] * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lrs[layer][connection]['bias'] * gradients[layer][connection]['bias']
    return params

def update_lr(gradients_0, gradients_1, lrs, hlr):
    for layer in lrs:
        for connection in lrs[layer]:

            lrs[layer][connection]['weights'] += hlr * (
                gradients_1[layer][connection]['weights'].flatten() @ gradients_0[layer][connection]['weights'].flatten()
            )

            lrs[layer][connection]['bias'] += hlr * (
                gradients_1[layer][connection]['bias'].flatten() @ gradients_0[layer][connection]['bias'].flatten()
            )

    return lrs



    # lr += hlr * (gradients_1 @ gradients_0)
    # return lr

def response(params, inputs = None, hps = None):
    return np.argmax(
        forward(params, inputs = inputs, hps = hps)[-1],
        axis = 1
    )


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import utils 
    # np.random.seed(0)
    np.seterr('raise')

    data = utils.organize_data_from_txt('iris.csv')

    sigmoid = lambda x:  1 / (1 + np.exp(-x))
    sigmoid_deriv = lambda x:  sigmoid(x) * (1 - sigmoid(x))

    hps = {
        'hyper_learning_rate': .000000005,  # <-- learning rate for hypergradient descent
        # 'learning_rate': .05,  # <-- learning rate
        'weight_range': [-3, 3],  # <-- weight range
        'num_hidden_nodes': 20,

        'hidden_activation': np.tanh,
        # 'hidden_activation_deriv': sigmoid_deriv,

        'output_activation': utils.softmax, # <-- linear output function
        # 'output_activation_deriv': lambda x: utils.softmax_deriv(x), # <-- derivative of linear output function
    
        'learning_rates': {
            'input': {
                'hidden': {
                    'weights': .05,
                    'bias': .05,
                },
            },
            'hidden': {
                'output': {
                    'weights': .05,
                    'bias': .05,
                } 
            },
        }
    }

    params = build_params(
        data['inputs'].shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        data['categories'].shape[0],
        weight_range = hps['weight_range']
    )
    num_epochs = 100

    print('loss initially: ', loss(params, inputs = data['inputs'], targets = data['one_hot_targets'], hps = hps))
    
    # for epoch in range(100):
    # for epoch in range(num_epochs):
    for epoch in range(1):
        gradients_0 = loss_grad(params, inputs = data['inputs'], targets = data['one_hot_targets'], hps = hps)
        params = update_params(params, gradients_0, hps['learning_rates'])
    

    for epoch in range(num_epochs - 1):
    # for epoch in range(100):
        gradients_1 = loss_grad(params, inputs = data['inputs'], targets = data['one_hot_targets'], hps = hps)
        hps['learning_rates'] = update_lr(gradients_0, gradients_1, hps['learning_rates'], hps['hyper_learning_rate'])
        params = update_params(params, gradients_1, hps['learning_rates'])

        gradients_0 = copy.deepcopy(gradients_1)


    print('loss after training: ', loss(params, inputs = data['inputs'], targets = data['one_hot_targets'], hps = hps))
    

    print(
        np.mean(
            np.equal(
                np.argmax(forward(params, inputs = data['inputs'], hps = hps)[-1], axis = 1),
                data['labels_indexed']
            )
        )
    )
