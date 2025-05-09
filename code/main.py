# Authors: https://github.com/meliketoy/wide-resnet.pytorch (the original training logic)
# and Jana Viktoria Kovacikova (refactoring and custom pruning logic)

from __future__ import print_function

import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn
import numpy as np
import random
import torch.nn.utils.prune as prune

import config as cf
from custom_pruning.utils import get_sparsity, print_sparsity_info
from custom_pruning.apply_pruning import apply_pruning
from acdc import acdc_train_network
from gradual_pruning import train_network_w_gradual_pruning
from custom_pruning.wide_resnet_layers_for_pruning import get_layers
from training_logic import (
    train_network,
    test_network,
    save_logs,
    load_checkpoint,
    create_savepoint,
    initialize_network,
    prepare_data,
)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_arguments() -> argparse.Namespace:
    """
    Parse the arguments from the command line.
    """
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10/100 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='The initial learning rate. The learning rate is later adjusted by the learning rate scheduler.')
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='Type of the network = [lenet/vggnet/resnet/wide-resnet]')
    parser.add_argument('--depth', default=28, type=int, help='The depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='The width of model')
    parser.add_argument('--dropout', default=0, type=float, help='The dropout rate')
    parser.add_argument('--dataset', default='cifar100', type=str, help='Dataset = [cifar10/cifar100]')
    parser.add_argument('--filename_to_save', type=str,
                        help='Filename to save the model. The default location is ./checkpoint/dataset_name/ and the specified filename gets appended to this path.')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint. This is to be used if you want to continue training from a saved model.')
    parser.add_argument('--resume_filepath', type=str,
                        help='The path to the checkpoint file to read the (pre-)trained model from. This must be specified if --resume or --testOnly is used.')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Only test the saved model.')
    parser.add_argument('--pruning_method', default='none', type=str,
                        help='The pruning method = [none/global_unstructured/block/block_iterative]')
    parser.add_argument('--sparsity', default=0.8, type=float,
                        help='The sparsity rate of the weights for pruning (0.0 - 1.0) - only used if pruning_method is not none')
    parser.add_argument('--block_criterion', default='max', type=str,
                        help='The criterion for selection of blocks to eliminate = [L1/L2/max/min]. It is used only if pruning_method is block or block_iterative.')
    parser.add_argument('--seed', default=21, type=int, help='Random seed')
    parser.add_argument('--acdc', action='store_true', help='Use the ACDC method for training')
    parser.add_argument('--remove_previous_pruning', action='store_true',
                        help='Remove previous pruning masks before training')
    parser.add_argument('--gradual_pruning', action='store_true', help='Use gradual pruning with parameters from config.py')
    return parser.parse_args()


def main():
    """
    Main function to train the neural network. It parses the arguments, prepares the data, initializes the network,
    loads the pretrained network (if specified), applies pruning (if specified), trains and/or tests the network.
    """
    args = parse_arguments()
    set_random_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'CPU'
    print("GPU device detected: " + device_name if use_cuda else "No GPU device detected: Using CPU")
    if use_cuda:
        cudnn.benchmark = True  # Not sure if this is necessary
        # Uncomment the following line, if you want to have deterministic results based on the seed.
        # torch.backends.cudnn.deterministic = True
        # Otherwise, the GPU may behave in a non-deterministic way.
    # I removed the parallelization part, as it turned out to be ineffective in our use case with 2 different GPUs

    trainset, testset, num_classes = prepare_data(args.dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cf.batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    if args.resume or args.testOnly:
        # Load the pretrained network
        net, start_epoch = load_checkpoint(args.resume_filepath)
    else:
        net = initialize_network(args.net_type, num_classes, args.depth, args.widen_factor, args.dropout)
        start_epoch = 1
    net.to(device)

    loss_criterion = nn.CrossEntropyLoss()

    # Test only mode
    # We always save the accuracy in the checkpoint though, so there is no need to use this option
    if args.testOnly:
        print("\nTesting the network")
        test_accuracy, loss = test_network(net, testloader, loss_criterion, device)
        print("| Test Result\tAcc@1: %.2f%%" % test_accuracy)
        return

    pruning_elapsed_time = 0
    pruning_layers = get_layers(net)
    if args.remove_previous_pruning:
        print("Turning the previous pruning off")
        for module in pruning_layers:
            prune.remove(module, 'weight')
    pruning_params = (args.pruning_method, args.sparsity, cf.block_size, args.block_criterion)
    if args.pruning_method != 'none' and not args.acdc and not args.gradual_pruning:
        print("\nApplying pruning")
        pruning_start_time = time.time()
        apply_pruning(net, *pruning_params)
        pruning_elapsed_time = time.time() - pruning_start_time
        print("\nSparsity after pruning - before training:")
        sparsity_info = get_sparsity(net, pruning_layers, pruning_params)
        print_sparsity_info(sparsity_info)

    savepoint = create_savepoint(args.dataset) + args.filename_to_save

    if args.acdc:
        training_elapsed_time, epoch_logs_list, pruning_elapsed_time = acdc_train_network(
            net, trainloader, testloader, loss_criterion, start_epoch, cf.num_epochs, args.lr, savepoint, device,
            pruning_params
        )
    elif args.gradual_pruning:
        training_elapsed_time, epoch_logs_list, pruning_elapsed_time = train_network_w_gradual_pruning(
            net, trainloader, testloader, loss_criterion, start_epoch, cf.num_epochs, args.lr, savepoint, device,
            pruning_params, n=cf.n, delta_t=cf.delta_t)
    else:
        training_elapsed_time, epoch_logs_list = train_network(
            net, trainloader, testloader, loss_criterion, start_epoch, cf.num_epochs, args.lr, savepoint, device
        )

    final_sparsity_info = get_sparsity(net, pruning_layers, pruning_params)
    print_sparsity_info(final_sparsity_info)
    save_logs(args, pruning_elapsed_time, training_elapsed_time, epoch_logs_list, device_name,
              log_file_path=savepoint + '_logs.json', final_global_sparsity=final_sparsity_info['Global sparsity'],
              final_sparsity_across_pruned_layers=final_sparsity_info['Total sparsity in pruned layers'])


if __name__ == '__main__':
    main()
