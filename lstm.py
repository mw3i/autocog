'''
Rough implementation of a Long-Short-Term-Memory Recurrent Network
~ barrowing heavily from the version made by HIPS (autograd team)

'''

import autograd.numpy as np 
from autograd import grad
from autograd.scipy.special import logsumexp

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def c(l): # <-- concatenate
    return np.concatenate(l, axis = 1)


def build_params(input_size, state_size, output_size):
    return [
        {
            'forget': {
                'w': np.random.normal(0, .01, [input_size + state_size , state_size]),
                'b': np.random.normal(0, .01, [1, state_size]),
            },
            'ingate': {
                'w': np.random.normal(0, .01, [input_size + state_size , state_size]),
                'b': np.random.normal(0, .01, [1, state_size]),
            },
            'change': {
                'w': np.random.normal(0, .01, [input_size + state_size , state_size]),
                'b': np.random.normal(0, .01, [1, state_size]),
            },
            'outgate': {
                'w': np.random.normal(0, .01, [input_size + state_size , state_size]),
                'b': np.random.normal(0, .01, [1, state_size]),
            },
            'predict': { 
                'w': np.random.normal(0, .01, [state_size, output_size]),
                'b': np.random.normal(0, .01, [1, output_size]),
            }
        },
        np.random.normal(0, .01, [1, state_size]), # <-- cell state
        np.random.normal(0, .01, [1, state_size]), # <-- hidden state
    ]



def forward_step(params, X = None, cell_state_0 = None, hid_state_0 = None):
    hid_state = np.repeat(hid_state_0, X.shape[0] - hid_state_0.shape[0] + 1, axis = 0)
    cell_state_1 = np.add(
        np.multiply( # <-- forget old info
            cell_state_0,
            sigmoid(c([X, hid_state]) @ params['forget']['w'] + params['forget']['b']), # <-- forget gate
        ),
        np.multiply( # <-- write new info
            sigmoid(c([X, hid_state]) @ params['ingate']['w'] + params['ingate']['b']), # <-- input gate
            np.tanh(c([X, hid_state]) @ params['change']['w'] + params['change']['b']), # <-- change gate
        )
    )

    hid_state_1 = np.multiply(
        sigmoid(c([X, hid_state]) @ params['outgate']['w']),
        # 1,
        np.tanh(cell_state_1)
    )

    return cell_state_1, hid_state_1


def forward_seq(params, X = None, init_cell_state = None, init_hidden_state = None):
    c, h = init_cell_state, init_hidden_state
    for t in range(X.shape[1]):
        c, h = forward_step(params, X = X[:,t:t+1], cell_state_0 = c, hid_state_0 = h)

    output = h @ params['predict']['w'] + params['predict']['b']
    return output


def loss(params, X = None, Y = None, init_cell_state = None, init_hidden_state = None):
    output = forward_seq(params, X = X, init_cell_state = init_cell_state, init_hidden_state = init_hidden_state)
    
    return np.sum(
        np.square(
            np.subtract(
                output,
                Y
            )
        )
    )# + np.sum(np.abs([np.sum(params[layer]['w']) for layer in params])) * .1 # <-- weight size regularization?

loss_grad = grad(loss)




'''
Train on toy problem (coming eventually...)
'''
if __name__ == '__main__':



    params, init_cell_state, init_hidden_state = build_params(3, 5, 2)


    # ##__Train
    # epochs = 10000
    # lr = .00005
    # for e in range(epochs):
    #     grads = loss_grad(params, X = X, Y = Y, init_cell_state = init_cell_state, init_hidden_state = init_hidden_state)
    #     for layer in grads:
    #         params[layer]['w'] -= lr * grads[layer]['w']
    #         params[layer]['b'] -= lr * grads[layer]['b']

    #     if e % 100 == 0: print(loss(params, X = X, Y = Y, init_cell_state = init_cell_state, init_hidden_state = init_hidden_state))




    # print('Real: \t\t', testY[:5, -1])

    # y = forward_seq(params, X = testX[:5,:], init_cell_state = init_cell_state, init_hidden_state = init_hidden_state)
    # print('Predicted: \t', np.round(y.T).astype(int))







