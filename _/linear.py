'''
Basic Linear Regression
3 Critical Functions:
    .forward(...) <-- generates output from input data
    .loss(...) <-- calculates success at predicting class labels
    .loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    .build(...) <-- generates a dictionary of random parameters (connections)
    utils.update_params(...) <-- updates param weights based on gradients provided
'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad

## int requirements
import utils


## produces model outputs
def forward(params, inputs = None, hps = None):
    output_activation = np.add(
            np.matmul(
                inputs,
                params['input']['output']['weights'],
            ),
            params['input']['output']['bias'],
        )
    return [output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs = inputs, hps = hps),
                targets
            )
        )
    )


## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_outcome_variables):
    '''
    num_features <-- (numeric) number of feature in the dataset
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'output': {
                'weights': np.zeros([num_features, num_outcome_variables]),
                'bias': np.zeros([1, num_outcome_variables]),
            }
        }
    }


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
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

    outcome_variables = np.array([
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
        'lr': .01,  # <-- learning rate
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        outcome_variables.shape[1]
    )

    num_epochs = 100

    print('loss initially: ', loss(params, inputs = inputs, targets = outcome_variables, hps = hps))
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = outcome_variables, hps = hps)
        params = utils.update_params(params, gradients, hps['lr'])
    print('loss after training: ', loss(params, inputs = inputs, targets = outcome_variables, hps = hps))
    
