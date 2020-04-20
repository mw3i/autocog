'''
based off of / sometimes directly copied from: 
    + kwj2104's example @ https://github.com/kwj2104/Simple-Variational-Autoencoder
    + HIPS group @ https://github.com/HIPS/autograd/blob/master/examples/variational_autoencoder.py

'''
import autograd.numpy as np 
from autograd import grad
import autograd.scipy.stats.norm as norm


# taken directly from: HIPS @ https://github.com/HIPS/autograd/blob/master/examples/variational_autoencoder.py
def diag_gaussian_log_density(x, mu, log_std):
    return np.sum(norm.logpdf(x, mu, np.exp(log_std)), axis=-1)


def forward(flattened_inputs, params, hps):

    ##__Encoder (push signal to latent layer)
    hidden1_activation = hps['hidden1_activation'](
        flattened_inputs @ params['input']['hidden1']['weights'] + params['input']['hidden1']['bias']
    )

    latent_means_activation = hps['latent_activation'](
        hidden1_activation @ params['hidden1']['latent_means']['weights'] + params['hidden1']['latent_means']['bias']
    )

    latent_vars_activation = hps['latent_activation'](
        hidden1_activation @ params['hidden1']['latent_vars']['weights'] + params['hidden1']['latent_vars']['bias']
    )

    ##__Generate Sample from Gaussian (reparameterization trick)
    latent_sample = latent_means_activation + np.exp(latent_vars_activation * .5) * np.random.standard_normal(size=(flattened_inputs.shape[0], hps['latent_nodes']))


    ##__Decoder (push latent sample to output layer)
    hidden3_activation = hps['hidden3_activation'](
        latent_sample @ params['latent']['hidden3']['weights'] + params['latent']['hidden3']['bias']
    )

    output_activation = hps['output_activation'](
        hidden3_activation @ params['hidden3']['output']['weights'] + params['hidden3']['output']['bias']
    )

    return latent_means_activation, latent_vars_activation, latent_sample, output_activation


epsilon = 10e-8 # <-- i think this prevents NaNs (not sure; copying off of kwj2104 here)
def loss(params, flattened_inputs = None, hps = None):

    latent_means_activation, latent_vars_activation, latent_sample, output_activation = forward(flattened_inputs, params, hps)

    recon_loss = np.sum(-flattened_inputs * np.log(output_activation + epsilon) - (1 - flattened_inputs) * np.log(1 - output_activation + epsilon))
    kl_div = -.5 * np.sum(1 + latent_vars_activation - np.square(latent_means_activation) - np.exp(latent_vars_activation))

    combined_loss = (recon_loss + kl_div) / flattened_inputs.shape[0]

    return combined_loss

loss_grad = grad(loss)



def gen_params(input_dims, hps):
    return {
        'input': {
            'hidden1': {
                'weights': np.random.randn(input_dims, hps['hidden1_nodes']) * np.sqrt(2 / input_dims), # <-- xavier initialization
                'bias': np.zeros([1,hps['hidden1_nodes']]),
            },
        },
        'hidden1': {
            'latent_means': {
                'weights': np.random.randn(hps['hidden1_nodes'], hps['latent_nodes']) * np.sqrt(2 / hps['hidden1_nodes']), # <-- xavier initialization
                'bias': np.zeros([1,hps['latent_nodes']]),
            },
            'latent_vars': {
                'weights': np.random.randn(hps['hidden1_nodes'], hps['latent_nodes']) * np.sqrt(2 / hps['hidden1_nodes']), # <-- xavier initialization
                'bias': np.zeros([1,hps['latent_nodes']]),
            },
        },
        'latent': {
            'hidden3': {
                'weights': np.random.randn(hps['latent_nodes'], hps['hidden3_nodes']) * np.sqrt(2 / hps['latent_nodes']), # <-- xavier initialization
                'bias': np.zeros([1,hps['hidden3_nodes']]),
            },
        },
        'hidden3': {
            'output': {
                'weights': np.random.randn(hps['hidden3_nodes'], input_dims) * np.sqrt(2 / hps['hidden3_nodes']), # <-- xavier initialization
                'bias': np.zeros([1,input_dims]),
            },
        },
    }


def gen_sample(params, latent_state):
    hidden3_activation = hps['hidden3_activation'](
        latent_state @ params['latent']['hidden3']['weights'] + params['latent']['hidden3']['bias']
    )

    output_activation = hps['output_activation'](
        hidden3_activation @ params['hidden3']['output']['weights'] + params['hidden3']['output']['bias']
    )

    return output_activation

def gen_samples(params, num_samples):
    sample = np.random.randn(num_samples, params['hidden1']['latent_means']['weights'].shape[1])

    hidden3_activation = hps['hidden3_activation'](
        sample @ params['latent']['hidden3']['weights'] + params['latent']['hidden3']['bias']
    )

    output_activation = hps['output_activation'](
        hidden3_activation @ params['hidden3']['output']['weights'] + params['hidden3']['output']['bias']
    )

    return output_activation


if __name__ == '__main__':
    import pickle

    from autograd.misc.optimizers import adam

    import utils
    from _ import vae_utils
    
    np.set_printoptions(linewidth=10000)

    stim, labels = vae_utils.load_shj()


    hps = {
        'hidden1_nodes': 25, # <-- first encoding step
        'latent_nodes': 3, # <-- assumed latent variables in probabilistic model
        'hidden3_nodes': 25, # <-- first decoding step

        'hidden1_activation': utils.relu,
        'latent_activation': lambda x: x, # <-- linear
        'hidden3_activation': utils.relu,
        'output_activation': utils.sigmoid,

        'lr': .01,
    }

    num_dims = 49 # <-- 7 times 7 (flattened image)
    params = gen_params(num_dims,hps)


    ## flatten 2d images into 1d vectors
    flattened_inputs = stim.reshape(stim.shape[0],-1) # <--  doing the reshaping before the code for efficiency purposes

    # ##__Manual gradient descent
    # for e in range(1000):
    #     if e % 10 == 0:
    #         print('loss: ', loss(params, flattened_inputs = flattened_inputs, hps = hps))
    #     gradients = loss_grad(params, flattened_inputs = flattened_inputs, hps = hps)

    #     # upd params
    #     params['input']['hidden1']['weights'] -= hps['lr'] * gradients['input']['hidden1']['weights']
    #     params['input']['hidden1']['bias'] -= hps['lr'] * gradients['input']['hidden1']['bias']

    #     params['hidden1']['latent_means']['weights'] -= hps['lr'] * gradients['hidden1']['latent_means']['weights']
    #     params['hidden1']['latent_means']['bias'] -= hps['lr'] * gradients['hidden1']['latent_means']['bias']
    #     params['hidden1']['latent_vars']['weights'] -= hps['lr'] * gradients['hidden1']['latent_vars']['weights']
    #     params['hidden1']['latent_vars']['bias'] -= hps['lr'] * gradients['hidden1']['latent_vars']['bias']
    
    #     params['latent']['hidden3']['weights'] -= hps['lr'] * gradients['latent']['hidden3']['weights']
    #     params['latent']['hidden3']['bias'] -= hps['lr'] * gradients['latent']['hidden3']['bias']
    
    #     params['hidden3']['output']['weights'] -= hps['lr'] * gradients['hidden3']['output']['weights']
    #     params['hidden3']['output']['bias'] -= hps['lr'] * gradients['hidden3']['output']['bias']


    # ##__Autograd Convenience Function 
    # def objective(params, iter_):
    #     return loss(params, flattened_inputs = flattened_inputs, hps = hps)
    # objective_grad = grad(objective)

    # print('before:', loss(params, flattened_inputs = flattened_inputs, hps = hps))
    # params = adam(
    #     objective_grad,
    #     params,
    #     step_size = .1,
    #     num_iters = 1000,
    # )
    # print('after:', loss(params, flattened_inputs = flattened_inputs, hps = hps))

    # print('Finished Training.\n-------------------------\n')

    # ## Save
    # with open('params.pckl', 'wb') as file:
    #     pickle.dump(params, file)



    # ----------------


    with open('params.pckl', 'rb') as file:
        params = pickle.load(file)



    ## plot single sample
    # sample = gen_sample(params, np.array([[0,0,0]])).reshape(7,7)
    # vae_utils.plot_sample(sample, 'sample.png', training_stim = stim)


    ## plot random samples
    num_samples = 50
    samples = gen_samples(params, num_samples)
    samples = samples.reshape(-1,7,7)
    vae_utils.plot_samples(samples, 'samples.png', training_stim = stim)

    ## plot grid
    samples = []
    mn = -2
    mx = 2
    grid_size = 8
    for x, g in enumerate(np.linspace(mn,mx,grid_size)):
        for y, gg in enumerate(np.linspace(mn,mx,grid_size)):
            samples.append(
                gen_sample(
                    params, 
                    np.array(
                        [[g,gg,0]]
                    )
                ).reshape(-1,7,7)[0]
            )
    vae_utils.plot_samples(samples, 'grid.png', training_stim = stim)
