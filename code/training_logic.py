# Authors: https://github.com/meliketoy/wide-resnet.pytorch (the original training logic)
# and Jana Viktoria Kovacikova (refactoring and custom pruning logic)

import os
import sys

import torch
import datetime
import time
import torch.nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import json

import config as cf
from networks import LeNet, VGG, ResNet, Wide_ResNet, conv_init


def prepare_data(dataset_name: str) -> (torch.utils.data.Dataset, torch.utils.data.Dataset, int):
    """
    Load the data based on dataset_name and apply meanstd transformation.

    :param dataset_name: name of the dataset (cifar10 or cifar100)
    :return: train_set, test_set, nr_classes (number of classes in the dataset)
    """
    print('\nData Preparation')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset_name], cf.std[dataset_name])
    ])  # meanstd transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[dataset_name], cf.std[dataset_name])
    ])

    nr_classes = 10
    if dataset_name == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    elif dataset_name == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
        nr_classes = 100
    else:
        raise ValueError("Invalid dataset name! Use 'cifar10' or 'cifar100'!")

    return train_set, test_set, nr_classes


def initialize_network(net_type: str, nr_classes: int, depth: int, widen_factor: int, dropout: float) -> torch.nn.Module:
    """
    Initialize the neural network based on the net_type and other parameters.

    :param net_type: type of the network (lenet, vggnet, resnet, wide-resnet)
    :param nr_classes: number of classes in the dataset (10 or 100)
    :param depth: depth of the network
    :param widen_factor: width of the network
    :param dropout: dropout rate
    :return: network: the initialized network
    """
    print('\nInitializing network [' + net_type + ']...')
    if net_type == 'lenet':
        network = LeNet(nr_classes)
    elif net_type == 'vggnet':
        network = VGG(depth, nr_classes)
    elif net_type == 'resnet':
        network = ResNet(depth, nr_classes)
    elif net_type == 'wide-resnet':
        network = Wide_ResNet(depth, widen_factor, dropout, nr_classes)
    else:
        raise ValueError('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')

    network.apply(conv_init)
    return network


def load_checkpoint(checkpoint_filepath: str) -> (torch.nn.Module, int):
    """
    Load the pretrained network from the checkpoint file.

    :param checkpoint_filepath: path to the checkpoint file
    :return: network, starting_epoch
    """
    print('\nLoading the pretrained network...')
    # Check if the path to the checkpoint exists
    assert os.path.isfile(checkpoint_filepath), 'Error: No checkpoint file found!'
    # In PyTorch 2.6, the default value of the `weights_only` argument in `torch.load` changed from `False` to `True`
    checkpoint = torch.load(checkpoint_filepath, weights_only=False)
    network = checkpoint['net']
    # We continue the training from the next epoch
    starting_epoch = checkpoint['epoch'] + 1
    print(f"| Accuracy of the loaded model: {checkpoint['acc']:.2f}%")
    return network, starting_epoch


def train_one_epoch(neural_network: torch.nn.Module, train_loader: torch.utils.data.DataLoader,
                    loss_criterion: torch.nn.modules.loss, device: torch.device, optimizer: optim.Optimizer) -> float:
    """
    Train the neural network for one epoch.

    :param neural_network: the neural network to train
    :param train_loader: the data loader for the training set
    :param loss_criterion: the loss function
    :param device: the device to use (CPU or GPU)
    :param optimizer: the optimizer to use

    :return: epoch_loss: the sum of batch losses for the epoch
    """
    neural_network.train()
    correct, total = 0, 0
    epoch_loss = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = neural_network(inputs)  # Forward Propagation
        loss = loss_criterion(outputs, targets)  # Loss
        epoch_loss += loss.item()
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    sys.stdout.write('\r')
    sys.stdout.write(f'\r| Batch Iter[{batch_idx + 1}/{len(train_loader)}]\t Last Batch Loss: {loss.item():.4f} Acc@1: {100. * correct / total:.3f}% \n')
    sys.stdout.flush()

    return epoch_loss


def create_savepoint(dataset_name: str) -> str:
    """
    Create a directory for saving the checkpoints (if not exists).

    :param dataset_name: name of the dataset
    :return: save_point: the path to the directory for saving the checkpoints
    """
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    save_point = './checkpoint/' + dataset_name + os.sep
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    return save_point


def save_checkpoint(neural_network: torch.nn.Module, accuracy: float, epoch_number: int, elapsed_time: float,
                    file_path: str) -> None:
    """
    Save the checkpoint of the neural network to file_path.
    """
    state = {
        'net': neural_network,
        'acc': accuracy,
        'epoch': epoch_number,
        'elapsed_time': elapsed_time,
    }
    torch.save(state, file_path)


def train_network(neural_network, train_loader, test_loader, loss_criterion, start_epoch, num_epochs,
                  initial_learning_rate, file_path_for_checkpoint, device):
    """
    Train the neural network for the specified number of epochs.
    The model with the best validation accuracy and the model from the last epoch are saved.
    """
    print('\nTraining model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(initial_learning_rate))
    print('| Optimizer = SGD')

    best_acc, valid_accuracy, elapsed_time = 0, 0, 0
    last_epoch = start_epoch + num_epochs - 1
    logging_list = []
    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):

        current_learning_rate = cf.learning_rate(initial_learning_rate, epoch)
        # The SGD optimizer is used
        optimizer = optim.SGD(neural_network.parameters(), lr=current_learning_rate, momentum=0.9, weight_decay=5e-4)
        print(f"\nTraining epoch: {epoch} / {last_epoch}, LR={current_learning_rate:.4f}")
        epoch_start_time = time.time()
        epoch_train_loss = train_one_epoch(neural_network, train_loader, loss_criterion, device, optimizer)
        # Validation step
        valid_accuracy, valid_loss = test_network(neural_network, test_loader, loss_criterion, device)
        epoch_time = time.time() - epoch_start_time
        print(f"| Validation Epoch #{epoch}\tLoss: {valid_loss:.4f} Acc@1: {valid_accuracy:.2f}%\t Epoch time: {epoch_time:.2f}s")
        elapsed_time = time.time() - start_time
        print(f'| Elapsed time: {datetime.timedelta(seconds=elapsed_time)}')

        if valid_accuracy > best_acc:
            print('| Saving the best model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time, file_path_for_checkpoint+'_bestepoch.t7')
            best_acc = valid_accuracy
        if epoch == last_epoch:
            print('| Saving the last model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time, file_path_for_checkpoint+'_lastepoch.t7')

        logging_list.append({'epoch': epoch, 'learning_rate': current_learning_rate, 'valid_accuracy': valid_accuracy,
                             'valid_loss': valid_loss, 'epoch_time': epoch_time, 'epoch loss': epoch_train_loss})

    print(f"\nBest Model Accuracy: {best_acc:.2f}%")
    print(f"\nFinal Model Accuracy in Last Epoch: {valid_accuracy:.2f}%")

    return elapsed_time, logging_list


def test_network(neural_network, test_loader, loss_criterion, device):
    """
    Test the neural network.
    """
    neural_network.eval()
    test_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = neural_network(inputs)
            loss = loss_criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        acc = 100. * correct / total

    # In the original script, they return the loss, but I decided to return the test_loss
    return float(acc), test_loss


def save_logs(args, pruning_time, elapsed_time, epoch_logs_list, device_name, log_file_path, final_global_sparsity,
              final_sparsity_across_pruned_layers):
    logs_data = {
        'device': device_name,
        # Each arg separately
        'initial_learning_rate': args.lr,
        'net_type': args.net_type,
        'depth': args.depth,
        'widen_factor': args.widen_factor,
        'dropout': args.dropout,
        'dataset': args.dataset,
        'checkpoint_filename': args.filename_to_save,
        'seed': args.seed,
        'resume': args.resume,
        'resumed_from_file': args.resume_filepath,
        'testOnly': args.testOnly,
        'pruning_method': args.pruning_method,
        'sparsity': args.sparsity,
        'block_criterion': args.block_criterion,
        'acdc': args.acdc,
        'pruning_time': pruning_time,
        # Config parameters
        'block_size': cf.block_size,
        'batch_size': cf.batch_size,
        'num_epochs': cf.num_epochs,
        # Training logs
        'epochs': epoch_logs_list,
        'elapsed_time_seconds': elapsed_time,
        'final_global_sparsity': final_global_sparsity,
        'final_sparsity_across_pruned_layers': final_sparsity_across_pruned_layers
    }

    with open(log_file_path, mode="w") as file:
        json.dump(logs_data, file, indent=4)
