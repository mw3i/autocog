'''
'''

## ext requirements
import autograd.numpy as np 
from autograd import grad

def softmax(x):
    x -= np.max(x)
    return (np.exp(x).T / np.sum(np.exp(x),axis=1)).T



def forward(params, inputs = None, hps = None):
    hidden_activation = hps['hidden_activation'](
        np.add(
            np.multiply(
                inputs,
                params['input']['hidden']['weights'],
            ),
            params['input']['hidden']['bias'],
        )
    )

    hidden_activation = np.exp(hidden_activation) / np.exp(hidden_activation).sum()

    channel_activations = hps['output_activation'](
        np.add(
            np.matmul(
                hidden_activation,
                params['hidden']['output']['weights'],
            ),
            params['hidden']['output']['bias'],
        )
    ) 

    return [hidden_activation, channel_activations]


## sum squared error loss function
def loss(params, inputs = None, targets = None, hps = None):
    return np.sum(
        np.square(
            np.subtract(
                forward(params, inputs = inputs, hps = hps)[-1],
                targets,
            )
        )
    )


## optimization function
loss_grad = grad(loss)


# - - - - - - - - - - - - - - - - - -


def build_params(num_features, num_categories):
    '''
    num_features <-- (numeric) number of feature in the dataset
    num_hidden_nodes <-- (numeric)
    num_categories <-- number of category channels to make
    weight_range = [-.1,.1] <-- (list of numeric)
    '''
    return {
        'input': {
            'hidden': {
                'weights': np.full([1, num_features], 1.),
                'bias': np.full([1, num_features], .5),
            },
        },
        'hidden': {
            'output': {
                'weights': np.full([num_features, num_categories], .3),
                'bias': np.full([1, num_categories], .5),
            } 
        },
    }


def update_params(params, gradients, lr0, lr1):
    params['input']['hidden']['weights'] -= lr0 * gradients['input']['hidden']['weights']
    params['input']['hidden']['bias'] -= lr0 * gradients['input']['hidden']['bias']

    params['hidden']['output']['weights'] -= lr1 * gradients['hidden']['output']['weights']
    params['hidden']['output']['bias'] -= lr1 * gradients['hidden']['output']['bias']

    return params



# - - - - - - - - - - - - - - - - - -



if __name__ == '__main__':
    import utils

    inputs = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],

    ])

    labels = [
        ['A','A','A','A', 'B','B','B','B'], # <-- type 1
        ['A','A','B','B', 'B','B','A','A'], # <-- type 2
        ['A','A','B','A', 'A','B','B','B'], # <-- type 3
        ['A','A','A','B', 'A','B','B','B'], # <-- type 4
        ['B','A','A','A', 'A','B','B','B'], # <-- type 5
        ['A','B','B','A', 'B','A','A','B'], # <-- type 6
    ]

    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}

    hps = {
        'lr0': 1.5,  # <-- learning rate rule
        'lr1': 1.5,  # <-- learning rate association

        # 'hidden_activation': lambda x: 1 / (1 + np.exp(x)), # <-- sigmoid
        'hidden_activation': lambda x: x, # <-- sigmoid
        # 'output_activation': lambda x: 1/ (1 + np.exp(x)),
        'output_activation': softmax,
    }

    learning_curves = []
    for dataset_labels in labels:
    
        labels_indexed = [idx_map[label] for label in dataset_labels]
        one_hot_targets = np.eye(len(categories))[labels_indexed]

        params = build_params(
            inputs.shape[1],  # <-- num features
            len(categories),
        )
        
        num_epochs = 20

        presentation = np.arange(inputs.shape[0])

        dataset_hist = []
        for epoch in range(num_epochs):
            np.random.shuffle(presentation)

            epoch_hist = []
            for i in presentation:

                epoch_hist.append(forward(params, inputs = inputs[i:i+1,:], hps = hps)[-1][0,labels_indexed[i]])

                gradients = loss_grad(params, inputs = inputs[i:i+1,:], targets = one_hot_targets[i:i+1,:], hps = hps)
                params = update_params(params, gradients, hps['lr0'], hps['lr1'])

            dataset_hist.append(np.mean(epoch_hist))

        learning_curves.append(dataset_hist)

    import matplotlib.pyplot as plt 
    for d, dataset in enumerate(['type1','type2','type3','type4','type5','type6']):
        plt.plot(learning_curves[d], label = dataset)
    plt.ylim([0,1.1])
    plt.legend()
    plt.savefig('test.png')


