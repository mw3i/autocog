'''
maybe a mixture of rbfs? related to gaussian mixture models? i dnt actually know
'''
## std lib

## ext requirements
import autograd.numpy as np 
from autograd import grad

softmax = lambda x: np.exp(x) / np.sum(np.exp(x))

def forward(params, inputs = None, hps = None):
    hidden_activation = np.array([
        np.exp(
            -np.matmul((
                np.subtract(
                    inputs,
                    params['input']['hidden']['bias'][:,h]
                ) * np.array([params['attn'],1 - params['attn']])) ** 2,
                params['input']['hidden']['weights'][:,h],
            )
        ) for h in range(params['input']['hidden']['weights'].shape[1])
    ]).T

    output_activation = hps['output_activation'](
        hidden_activation @ params['hidden']['output']['weights']
    )

    return [hidden_activation, output_activation]


## logistic loss function
def loss(params, inputs = None, targets = None, hps = None):
    # return np.sum(
    #     np.square(
    #         np.subtract(
    #             forward(params, inputs = inputs, hps = hps)[-1],
    #             targets
    #         )
    #     )
    # )
    return -np.sum(
        targets * np.log(forward(params, inputs = inputs, hps = hps)[-1]),
    ) / targets.shape[0]

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
                'weights': np.full([num_features, num_hidden_nodes],10.0),
                'bias': np.random.normal(*weight_range, [num_features, num_hidden_nodes]),
            },
        },
        'hidden': {
            'output': {
                'weights': np.random.normal(*weight_range, [num_hidden_nodes, len(categories)]),
                'bias': np.random.normal(*weight_range, [1, len(categories)]),
            },
        },
        'attn': .5,
    }


def update_params(params, gradients, lr):
    params['input']['hidden']['weights'] -= lr * gradients['input']['hidden']['weights']
    params['input']['hidden']['bias'] -= lr * gradients['input']['hidden']['bias']

    params['hidden']['output']['weights'] -= lr * gradients['hidden']['output']['weights']

    # params['attn'] -= lr * gradients['attn']


    # for layer in params:
        # for connection in params[layer]:
            # params[layer][connection]['weights'] -= lr * gradients[layer][connection]['weights']
            # if layer == 'input': params[layer][connection]['bias'] -= lr * gradients[layer][connection]['bias']
    return params


# - - - - - - - - - - - - - - - - - -


if __name__ == '__main__':
    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    cmap_ = 'binary'

    fig = plt.figure(
        figsize = [8,2]
    )

    gs = GridSpec(1,4)

    cv = .001
    inputs = np.concatenate([
        np.random.multivariate_normal(
            [.2,.2], 
            [
                [.005,cv],
                [cv,.005],
            ],
            [50]
        ),
        np.random.multivariate_normal(
            [.8,-.2], 
            [
                [.005,cv],
                [cv,.005],
            ],
            [50]
        ),
        np.random.multivariate_normal(
            [.8,.8], 
            [
                [.005,cv],
                [cv,.005],
            ],
            [50]
        )
    ])
    labels = [0] * 100 + [1] * 50
    cm = {0:'orange',1:'blue'}

    data_ax = plt.subplot(gs[0,0])
    data_ax.scatter(
        *inputs.T,
        c = [cm[l] for l in labels],
    )

    # data_ax.set_ylim([0,1]); data_ax.set_xlim([0,1])
    data_ax.set_yticks([]); data_ax.set_xticks([])





    categories = np.unique(labels)
    idx_map = {category: idx for category, idx in zip(categories, range(len(categories)))}
    
    labels_indexed = [idx_map[label] for label in labels]
    one_hot_targets = np.eye(len(categories))[labels_indexed]

    hps = {
        'lr': .05,  # <-- learning rate
        'wr': [.5, .1],  # <-- weight range
        'num_hidden_nodes': 10,

        'output_activation': lambda x: softmax(x)
        # 'output_activation': lambda x: 1 / (1 + np.exp(-x)),
        # 'output_activation': lambda x: np.exp(-(x ** 2)),
    }

    params = build_params(
        inputs.shape[1],  # <-- num features
        hps['num_hidden_nodes'],
        categories,
        weight_range = hps['wr']
    )
    # params['input']['hidden']['bias'] = np.array([
    #     [.2,.5,.8],
    #     [.2,.5,.8],
    # ])

    num_epochs = 500

    print('loss initially: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))

    for epoch in range(num_epochs):
        gradients = loss_grad(params, inputs = inputs, targets = one_hot_targets, hps = hps)
        params = update_params(params, gradients, hps['lr'])
    print('loss after training: ', loss(params, inputs = inputs, targets = one_hot_targets, hps = hps))
    print('model accuracy:', 
        np.mean(
            np.equal(
                np.argmax(
                    forward(params, inputs = inputs, hps = hps)[-1],
                    axis = 1,
                ),
                labels
            )
        )
    )

    g = 20
    m1, m2 = [-1,2]
    mesh = np.array(np.meshgrid(np.linspace(m1,m2,g), np.linspace(m1,m2,g))).reshape(2, g*g).T

    pred_ax = plt.subplot(gs[0,1])
    pred_ax.imshow(
        np.flip(forward(params, inputs = mesh, hps = hps)[-1][:,0].reshape(g,g), axis = 0),
        extent = [m1,m2,m1,m2],
        cmap = 'binary',
    )
    pred_ax.scatter(
        *params['input']['hidden']['bias'],
        s = np.abs(params['hidden']['output']['weights'][:,0]) * 200,
        c = ['red' if w < 0 else 'black' for w in params['hidden']['output']['weights'][:,0]]
    )
    pred_ax.scatter(
        *inputs.T,
        c = [cm[l] for l in labels], alpha = .05
    )

    pred_ax.set_xticks([]);pred_ax.set_yticks([])

    # - - - - 

    pred_ax = plt.subplot(gs[0,2])
    pred_ax.imshow(
        np.flip(forward(params, inputs = mesh, hps = hps)[-1][:,1].reshape(g,g), axis = 0),
        extent = [m1,m2,m1,m2],
        cmap = 'binary',
    )
    pred_ax.scatter(
        *params['input']['hidden']['bias'],
        s = np.abs(params['hidden']['output']['weights'][:,1]) * 200,
        c = ['red' if w < 0 else 'black' for w in params['hidden']['output']['weights'][:,0]]
    )
    pred_ax.scatter(
        *inputs.T,
        c = [cm[l] for l in labels], alpha = .05
    )
    pred_ax.set_xticks([]);pred_ax.set_yticks([])

    # - - - - 

    pred_ax = plt.subplot(gs[0,3])
    pred_ax.imshow(
        np.flip((np.abs(forward(params, inputs = mesh, hps = hps)[-1][:,1] - forward(params, inputs = mesh, hps = hps)[-1][:,0]).reshape(g,g)), axis = 0), # <-- stand in for model confidence
        # np.flip((forward(params, inputs = mesh, hps = hps)[-1] / forward(params, inputs = mesh, hps = hps)[-1].max(axis = 1, keepdims = True))[:,0].reshape(g,g), axis = 0),
        # np.flip(forward(params, inputs = mesh, hps = hps)[-1].argmax(axis = 1).reshape(g,g), axis = 0),
        # np.flip(forward(params, inputs = mesh, hps = hps)[-1].max(axis = 1).reshape(g,g), axis = 0),
        # np.flip(np.product(forward(params, inputs = mesh, hps = hps)[-1], axis = 1).reshape(g,g), axis = 0),
        # np.flip(np.sum(forward(params, inputs = mesh, hps = hps)[-1], axis = 1).reshape(g,g), axis = 0),
        extent = [m1,m2,m1,m2], cmap = cmap_
    )
    pred_ax.set_xticks([]);pred_ax.set_yticks([])


    plt.savefig('test.png')
    # plt.show()
