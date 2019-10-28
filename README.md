# autoCog

_Cognitive psychology models built with [numpy](http://www.numpy.org/) & [autograd](https://github.com/HIPS/autograd)_

#### Currently Implemented Models:

- Linear Regression
- Logistic Regression
- Multilayer Classifier
- Autoencoder
- DIVergent Autoencoder ([Kurtz, 2007](http://kurtzlab.psychology.binghamton.edu/publications/diva-pbr.pdf)) & Modifications
    - gendiscrim version
    - global average pooling version
    - shallow versions
    - detangler
- Multitasking Network
- ALCOVE ([Krushke, 1992](http://www.indiana.edu/~pcl/rgoldsto/courses/concepts/Kruschke1992.pdf))
    - ^ a little off from original model because the similarity function is non-differentiable at zero (i think)
- WARP (Kurtz & Silliman, 2018)

## Install
`git clone https://github.com/mwetzel7r/autoCog`

#### requirements:
- numpy
- autograd
- probably scipy eventually
- **optional**:
    - matplotlib (plotting)

## Usage
- models typically utilize the same basic functions:
    - `forward(...)`: activate model
    - `loss(...)`: cost function of model activation
    - `loss_grad(...)`: returns gradients to update model weights
- for most models, you update the parameters with the function:
    -`update_params(...)`

## Roadmap
- keep the modules very light weight and consistent
    - forward, loss, loss_grad functions in every model (that allow them)
    - only use numpy & autograd, and sometimes scipy
    - parallelization would be cool but id rather wait for the autograd team to deal with that
- want to include other cognitive models besides neural nets
    - covis
    - bayesian things
    - all the mclellend & team's models
    - convolutional & recurrent nets
    - neural turing machines & differentiable computers
        - ^ cuz i think they deserve being tested as model's of human cognition




