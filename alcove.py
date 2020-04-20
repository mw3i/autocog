'''
Attention Learned COVering Map ([Krushke 1992](http://www.indiana.edu/~pcl/rgoldsto/courses/concepts/Kruschke1992.pdf))
3 Critical Functions:
    forward(...) <-- generates output from input data
    loss(...) <-- calculates success at classifying the input data
    loss_grad(...) <-- calculates gradients for the loss function

Other Useful Functions:
    build(...) <-- generates a dictionary of parameters (connections)
    update_params(...) <-- updates param weights based on gradients provided

NOTE!!!! This model doesn't work exactly as described in Kruschke's original paper; primarily because the similarity equation using the minkowsky dist func isn't differentiable when the distance metric is >= 2
'''

## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad
from scipy import spatial
np.set_printoptions(suppress=True)

## int requirements
import utils

minfloat = np.finfo(np.double).tiny

def pdist(a1, a2, r, **kwargs):
    attention_weights = kwargs.get('attention_weights', np.ones([1,a1.shape[1]]) / a1.shape[1])

    # format inputs & exemplars for (i think vectorized) pairwise distance calculations
    a1_tiled = np.tile(a1, a2.shape[0]).reshape(a1.shape[0], a2.shape[0], a1.shape[1])
    a2_tiled = np.repeat([a2], a1.shape[0], axis=0)

    if hps['r'] > 1:
        # get attention-weighted pairwise distances
        distances = np.sum(
            np.multiply(
                attention_weights,
                np.abs(a1_tiled - a2_tiled) ** r
            ),
            axis = 2,
        )

        distances = np.power(
            np.where(
                distances > 0,
                distances,
                minfloat
            ),
            1/r
        )

    else:
        distances = np.sum(
            np.multiply(
                attention_weights,
                np.abs(a1_tiled - a2_tiled)
            ),
            axis = 2,
        )

    return distances

## produces model outputs
def forward(params, inputs = None, exemplars = None, hps = None):

    distances = pdist(inputs, exemplars, hps['r'], attention_weights = params['attention_weights'])

    # exemplar layer activations
    hidden_activation = np.exp(
        (-hps['c']) * distances
    )
    # class predictions (luce-choiced, or, softmaxed)
    output_activation = np.matmul(
            hidden_activation,
            params['association_weights']
        ).clip(-1.,1.)

    return [hidden_activation, output_activation]


## sum squared error loss function
def loss(params, inputs = None, exemplars = None, targets = None, hps = None):
    output_activation = forward(params, inputs = inputs, exemplars = exemplars, hps = hps)[-1]
    targets = (output_activation * targets).clip(1, np.inf) * targets # <-- humble teacher principle (performs max(1,t) func on correct category labels, and min(-1,t) on incorrect channels)

    return .5 * np.sum(
        np.square(
            np.subtract(
                output_activation,
                targets
            )
        )
    )

## optimization function
loss_grad = grad(loss)



def probabilities(params, inputs = None, exemplars = None, hps = None): # softmax
    output_activation = forward(params, inputs = inputs, exemplars = exemplars, hps = hps)[-1]
    return np.divide(
        np.exp(output_activation * hps['phi']),
      #---------#
        np.sum(
            np.exp(output_activation * hps['phi']), 
            axis=1, keepdims=True
        )
    )


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_exemplars, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_categories <-- (list) 
    '''
    return {
        'attention_weights': np.ones([1, num_features]) / num_features,
        'association_weights': np.zeros([num_exemplars, num_categories]),
    }



# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import copy 

    inputs = np.array([
        [.05,  .05],
        [.1, .1],
        [.1, .2],
        [.2, .1],
        [.2, .2],

        [1., 1.],
        [.9, .9],
        [.9, .8],
        [.8, .9],
        [.8, .8],
    ])

    exemplars = inputs

    labels = [
        'A','A','A','A','A',   'B','B','B','B','B',
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    labels_indexed = [idx_map[label] for label in labels]
    one_pos_targets = (np.eye(len(categories))[labels_indexed] * 2) - 1

    hps = {
        'c': 1,  # <-- specificity parameter
        'r': 2, # <-- distance metric
        'atlr': .2, # <-- attention learning rate
        'aslr': .1, # <-- association learning rate
        'phi': 4, # <-- response mapping parameter
    }

    init_params = build_params(
        inputs.shape[1],  # <-- num features
        exemplars.shape[0], # <-- num exemplars
        len(categories), # <-- num categories
    )
    
    num_epochs = 3


    print('Trial-Wise Training\n-----------')
    params = copy.deepcopy(init_params)
   
    # print('loss initially: ', loss(params, inputs = inputs, exemplars = exemplars, targets = one_pos_targets, hps = hps))
    for e in range(num_epochs):
        for i in range(inputs.shape[0]):
            gradients = loss_grad(params, inputs = inputs[i:i+1,:], exemplars = exemplars, targets = one_pos_targets[i:i+1,:], hps = hps)

            # update attn weights
            params['attention_weights'] -= (hps['atlr'] * gradients['attention_weights'])
            params['attention_weights'] = utils.relu(params['attention_weights'])

            # update association weights
            params['association_weights'] -= hps['aslr'] * gradients['association_weights']

    print('loss after training: ', loss(params, inputs = inputs[i:i+1,:], exemplars = exemplars, targets = one_pos_targets[i:i+1,:], hps = hps))
    print('________________\n\n')


    # print('Batch Training\n-----------')
    # params = copy.deepcopy(init_params)

    # print('loss initially: ', loss(params, inputs = inputs, exemplars = exemplars, targets = one_pos_targets, hps = hps))
    # for e in range(num_epochs):
    #     gradients = loss_grad(params, inputs = inputs, exemplars = exemplars, targets = one_pos_targets, hps = hps)

    #     # update attn weights
    #     params['attention_weights'] -= (hps['atlr'] * gradients['attention_weights'])
    #     params['attention_weights'] = utils.relu(params['attention_weights'])

    #     # update association weights
    #     params['association_weights'] -= hps['aslr'] * gradients['association_weights']

    # print('loss after training: ', loss(params, inputs = inputs, exemplars = exemplars, targets = one_pos_targets, hps = hps))
    # print('________________\n\n')


    # print(probabilities(params, inputs = inputs, exemplars = exemplars, hps = hps))

