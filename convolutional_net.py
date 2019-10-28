## std lib
import sys, os

## ext req
import autograd.numpy as np 
from autograd import grad
import autograd.scipy.signal as signal


## _ _ _ Get Model Output _ _ _ 
def forward(model, inputs = None, hps = None):
    conv_act = hps['c1_activation'](
        np.add(
            signal.convolve(
                inputs,
                model['input']['c1']['weights']
            ),
            model['input']['c1']['bias']
        )
    ).reshape(inputs.shape[0], -1) # <-- flatten final activations

    dense1_act = hps['d1_activation'](
        np.add(
            np.matmul(
                conv_act,
                model['c1']['d1']['weights']
            ),
            model['c1']['d1']['bias']
        )
    )

    output_act = hps['output_activation'](
        np.add(
            np.matmul(
                dense1_act,
                model['d1']['output']['weights']
            ),
            model['d1']['output']['bias']
        )
    )
    return output_act


## _ _ _ Cost Function _ _ _ 
# cross entropy loss function
def loss(params, inputs = None, targets = None, hps = None):
    model_output = forward(params, inputs = inputs, hps = hps)
    return -np.sum(
        targets * np.log(model_output),
    )

loss_grad = grad(loss)


## _ _ _ Build Model _ _ _ 
def build_params(input_dimensions, categories, hps):
    c1_size_flattened = np.prod([input_dimensions[-3], input_dimensions[-2]+2, input_dimensions[-1]+2, hps['c1_filters']])
    return {
        'input': {
            'c1': {
                'weights': np.random.uniform(*hps['wr'], [1, hps['c1_filters'], *hps['c1_filtersize']]), # <-- feature maps
                'bias': np.zeros([1, hps['c1_filters'], 1, 1])
            },
        },
        'c1': {
            'd1': {
                'weights': np.random.uniform(*hps['wr'], [c1_size_flattened, hps['d1_nodes']]),
                'bias': np.zeros([1,hps['d1_nodes']]),
            },
        },
        'd1': {
            'output': {
                'weights': np.random.uniform(*hps['wr'], [hps['d1_nodes'],len(categories)]),
                'bias': np.zeros([1,len(categories)]),
            },
        },
    }

def build_params_smart(input_dimensions, categories, hps):
    c1_size_flattened = np.prod([input_dimensions[-3], input_dimensions[-2]+2, input_dimensions[-1]+2, hps['c1_filters']])
    return {
        'input': {
            'c1': { # he et al 
                'weights': np.random.normal(0, 1, [1, hps['c1_filters'], *hps['c1_filtersize']]) * 2 * np.sqrt(2 / np.multiply(*hps['c1_filtersize'])), # <-- feature maps
                'bias': np.zeros([1, hps['c1_filters'], 1, 1])
            },
        },
        'c1': {
            'd1': { # xavier
                'weights': np.random.normal(0, 1, [c1_size_flattened, hps['d1_nodes']]) * np.sqrt(2 / (c1_size_flattened + hps['d1_nodes'])),
                'bias': np.zeros([1,hps['d1_nodes']]),
            },
        },
        'd1': {
            'output': {
                'weights': np.random.normal(0, 1, [hps['d1_nodes'],len(categories)]) * np.sqrt(2 / (hps['d1_nodes'] + len(categories))),
                'bias': np.zeros([1,len(categories)]),
            },
        },
    }

def update_params(params, gradients, lr):
    for layer in params:
        for connection in params[layer]:
            params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


if __name__ == '__main__':
    
    ## _ _ _ Load data _ _ _    
    ## generate made up images
    inputs = np.array([
        ## square small light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square large light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square small dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, 1., 1., 1., .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## square large dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, 1., 1., 1., 1., 1., .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond small light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond large light
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, 1., .2, .2, .2, 1., .0],
            
            [.0, .0, 1., .2, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond small dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],

        ## diamond large dark
        [[
            [.0, .0, .0, .0, .0, .0, .0],

            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, 1., .8, .8, .8, 1., .0],
            
            [.0, .0, 1., .8, 1., .0, .0],
            
            [.0, .0, .0, 1., .0, .0, .0],
            
            [.0, .0, .0, .0, .0, .0, .0],
        ]],
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

    ## _ _ _ Define Hyperparameters _ _ _ 
    hps = {
        'lr': .1, # <-- learning rate
        'wr': [-.1,.1], # <-- weight range
        'c1_activation': lambda x: x * (x > 1), # relu
        'd1_activation': lambda x: 1 / (1 + np.exp(-x)), # sigmoid
        'output_activation': lambda x: np.exp(x) / np.sum(np.exp(x), axis = 1, keepdims = True), # softmax

        'c1_filters': 6, # <-- conv layer
        'c1_filtersize': [3,3], # <-- dimensions of filters (scipy will call you out if the size doesn't work)
        'd1_nodes': 10, # <-- dense layer
    }

    ## _ _ _ Build Model _ _ _ 
    # params = build_params(
    #     inputs.shape, # <-- input dimensions
    #     categories,
    #     hps,
    # )
    params = build_params_smart(
        inputs.shape, # <-- input dimensions
        categories,
        hps,
    )

    ## _ _ _ Run MODEL _ _ _ 
    print('- - - RUN MODEL - - -')

    print('init loss:', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    init_maps = params['input']['c1']['weights'].copy()

    num_epochs = 10
    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    
    print('end loss:', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    print('- - - done - - - ')
