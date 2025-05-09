############### Pytorch CIFAR configuration file ###############
# Authors: https://github.com/meliketoy/wide-resnet.pytorch and Jana Viktoria Kovacikova
import math

num_epochs = 200
batch_size = 128  # It is used for training only. For testing, we use 100.
block_size = 8

mean = {
    'cifar10': (0.4914, 0.4822, 0.4465),
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar10': (0.2023, 0.1994, 0.2010),
    'cifar100': (0.2675, 0.2565, 0.2761),
}

# Only for cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# LR schedule (a)
# def learning_rate(init, epoch):
#     optim_factor = 0
#     if(epoch > 160):
#         optim_factor = 3
#     elif(epoch > 120):
#         optim_factor = 2
#     elif(epoch > 60):
#         optim_factor = 1
#     return init*math.pow(0.2, optim_factor)


# LR schedule (b)
def learning_rate(init, epoch):
    optim_factor = 0
    if 200 < epoch < 260:
        optim_factor = 1
    elif 260 <= epoch < 320:
        optim_factor = 2
    elif epoch > 160:
        optim_factor = 3
    elif epoch > 120:
        optim_factor = 2
    elif epoch > 60:
        optim_factor = 1
    return init*math.pow(0.2, optim_factor)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


# Settings for gradual pruning - only used when program called with --gradual_pruning in args
n = 49  # Number of pruning steps
delta_t = 1  # Increase pruning after delta_t epochs
initial_sparsity = 0.5  # The pruning to this initial sparsity will be applied prior to gradual pruning training
# - but only if the corresponding parameter is set to True in gradual_pruning function
