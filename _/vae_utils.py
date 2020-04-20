import autograd.numpy as np 

def load_shj():
    labels = np.array(
        [0,0,1,1,0,0,1,1]
    )

    ## Stim Dimensions (Symbolic)
        # 0 0 0
        # 0 1 0
        # 0 0 1
        # 0 1 1
        # 1 0 0
        # 1 1 0
        # 1 0 1
        # 1 1 1

    ## Actual Stimuli (Perceptual)
    stim = np.array([

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

    return [stim, labels]




# # # # # # # # # 
# 
#   Plotting Utils
# 
# # # # # # # # # 
clean = lambda ax: [ax.set_xticks([]), ax.set_yticks([])]

def plot_samples(samples, filename, training_stim = None):
    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    
    ##__Plot Samples
    num_samples = len(samples)

    if training_stim is None:
        num_training_stim = int(np.sqrt(len(samples)).round(0))
    else: 
        num_training_stim = training_stim.shape[0]

    fig = plt.figure(
        # figsize = []
    )
    row_width = num_training_stim
    gs = GridSpec(2 + num_samples // row_width + 1, row_width)

    # plot original
    if (training_stim is None) == False:
        for s in range(training_stim.shape[0]):
            ax = plt.subplot(gs[0, s % row_width])
            ax.imshow(
                training_stim[s,0,:,:],
                cmap = 'binary'
            )
            clean(ax)
            if s == 0: ax.set_ylabel('original')

    # plot divider
    ax = plt.subplot(gs[1,:])
    ax.imshow(np.ones([1,100]),cmap='binary',vmin=0,vmax=1)
    clean(ax)


    # plot samples
    for s in range(num_samples):
        ax = plt.subplot(gs[s // row_width + 2, s % row_width])
        ax.imshow(
            samples[s],
            cmap = 'binary'
        )
        clean(ax)

    plt.savefig(filename)
    plt.close()


def plot_sample(sample, filename, training_stim = None):
    import matplotlib.pyplot as plt 
    from matplotlib.gridspec import GridSpec
    
    ##__Plot Samples
    if training_stim is None:
        num_training_stim = 10
    else: 
        num_training_stim = training_stim.shape[0]

    fig = plt.figure(
        # figsize = []
    )
    row_width = num_training_stim
    gs = GridSpec(2 + num_training_stim + 1, num_training_stim)

    # plot original
    if (training_stim is None) == False:
        for s in range(training_stim.shape[0]):
            ax = plt.subplot(gs[0, s % row_width])
            ax.imshow(
                training_stim[s,0,:,:],
                cmap = 'binary'
            )
            clean(ax)
            if s == 0: ax.set_ylabel('original')

    # plot divider
    ax = plt.subplot(gs[1,:])
    ax.imshow(np.ones([1,100]),cmap='binary',vmin=0,vmax=1)
    clean(ax)


    # plot samples
    ax = plt.subplot(gs[2:,:])
    ax.imshow(
        sample,
        cmap = 'binary'
    )
    clean(ax)

    plt.savefig(filename)
    plt.close()




