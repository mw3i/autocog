import autograd.numpy as anp 
from autograd import grad


def forward(params, architecture = None, inputs = None, path = None):
    layer_activations = []
    
    ## first layer
    layer_activations.append(
        architecture['input']['activation'](
            inputs
        )
    )

    for l, layer in enumerate(path[1:]):
        layer_activations.append(
            architecture[layer]['activation'](
                anp.add(
                    anp.matmul(
                        layer_activations[-1],
                        params[path[l]][layer]['weights']
                    ),
                    params[path[l]][layer]['bias']
                )
            )
        )

    return layer_activations


def loss(params, architecture = None, inputs = None, targets = None, path = None):
    return anp.sum(
        anp.square(
            anp.subtract(
                forward(params, architecture = architecture, inputs = inputs, path = path)[-1],
                targets
            )
        )
    )


loss_grad = grad(loss)


def build_params(architecture):
    params = {}
    for layer in architecture:
        params[layer] = {}
        for connection in architecture[layer]['connections']:
            params[layer][connection] = {
                'weights': anp.random.uniform(*architecture[layer]['connections'][connection]['weight_range'], [architecture[layer]['nodes'], architecture[connection]['nodes']]),
                'bias': anp.random.uniform(*architecture[layer]['connections'][connection]['weight_range'], [1, architecture[connection]['nodes']]),
            }
    return params

def update_params(params, architecture, gradients, path):
    for l, layer in enumerate(path[:-1]):
        params[layer][path[l+1]]['weights'] -= architecture[layer]['connections'][path[l+1]]['learning_rate'] * gradients[layer][path[l+1]]['weights']
        params[layer][path[l+1]]['bias'] -= architecture[layer]['connections'][path[l+1]]['learning_rate'] * gradients[layer][path[l+1]]['bias']
    return params



## run test
if __name__ == '__main__':
    inputs = anp.random.normal(0,.4,[10,3])

    settings = {
        'learning_rate': .005,
        'weight_range': [-1,1],
    }

    architecture = {
        'input': {
            'nodes': inputs.shape[1],
            'activation': lambda x: x,
            'connections': {
                'hidden': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden': {
            'nodes': 5,
            'activation': anp.tanh,
            'connections': {
                'hidden2': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden2': {
            'nodes': 10,
            'activation': anp.tanh,
            'connections': {
                'hidden3': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden3': {
            'nodes': 10,
            'activation': anp.tanh,
            'connections': {
                'hidden4': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden4': {
            'nodes': 10,
            'activation': anp.tanh,
            'connections': {
                'hidden5': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden5': {
            'nodes': 10,
            'activation': anp.tanh,
            'connections': {
                'hidden6': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'hidden6': {
            'nodes': 10,
            'activation': anp.tanh,
            'connections': {
                'output': {
                    'learning_rate': settings['learning_rate'],
                    'weight_range': settings['weight_range'],
                }
            }
        },
        'output': {
            'nodes': inputs.shape[1],
            'activation': lambda x: x,
            'connections': {
            }
        }
    }

    params = build_params(architecture)


    path = list(architecture.keys())
    print('paht:', path)
    print('init loss:', loss(params, architecture = architecture, inputs = inputs, targets = inputs, path = path))
    for i in range(1000):
        params = update_params(
            params,
            architecture,
            loss_grad(params, architecture = architecture, inputs = inputs, targets = inputs, path = path),
            path
        )
    print('end loss:', loss(params, architecture = architecture, inputs = inputs, targets = inputs, path = path))



