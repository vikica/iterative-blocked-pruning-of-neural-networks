import time
import datetime
import torch.optim as optim

import config as cf
from custom_pruning.utils import get_sparsity, print_sparsity_info
from custom_pruning.apply_pruning import apply_pruning
from custom_pruning.wide_resnet_layers_for_pruning import get_layers
from training_logic import train_one_epoch, test_network, save_checkpoint


def check_params(last_epoch: int, n: int, delta_t: int, t0: int, initial_sparsity, goal_sparsity) -> None:
    sparsity_in_last_epoch = calculate_current_goal_sparsity(last_epoch, n, t0, delta_t, initial_sparsity,
                                                             goal_sparsity)
    if sparsity_in_last_epoch != goal_sparsity:
        raise ValueError(
            f"Invalid parameters! The calculated goal sparsity in the last epoch ({sparsity_in_last_epoch}) "
            f"does not match the specified goal sparsity ({goal_sparsity}). "
            f"Please adjust the parameters for gradual pruning. "
        )


def calculate_current_goal_sparsity(epoch: int, n: int, t0: int, delta_t: int, initial_sparsity: float,
                                    goal_sparsity: float) -> float:
    """
    Calculate the current goal sparsity based on the epoch number and the gradual pruning parameters.

    :param epoch: The current epoch number.
    :param n: The number of pruning steps.
    :param t0: The starting epoch for gradual pruning.
    :param delta_t: The number of epochs between pruning steps.
    :param initial_sparsity: The initial sparsity of the network.
    :param goal_sparsity: The target sparsity of the network.

    :return: The current goal sparsity.
    """
    return goal_sparsity + (initial_sparsity - goal_sparsity) * (1 - ((epoch - t0) / (n * delta_t)))


def train_network_w_gradual_pruning(neural_network, train_loader, test_loader, loss_criterion, start_epoch, num_epochs,
                                    initial_learning_rate, file_path_for_checkpoint, device, pruning_params, n,
                                    delta_t, use_config_initial_sparsity=False):
    """
    Train the neural network for the specified number of epochs, using gradual pruning schedule.
    The model with the best validation accuracy and the model from the last epoch are saved.

    :param neural_network: The neural network to train.
    :param train_loader: The data loader for the training set.
    :param test_loader: The data loader for the test set.
    :param loss_criterion: The loss criterion to use for training.
    :param start_epoch: The starting epoch number.
    :param num_epochs: The number of epochs for training.
    :param initial_learning_rate: The initial learning rate for training.
    :param file_path_for_checkpoint: The path to save the checkpoints.
    :param device: The device to use for training.
    :param pruning_params: The parameters for pruning the network.
                           Tuple of (pruning_method, sparsity, block_size, block_criterion).
    :param n: The number of pruning steps.
    :param delta_t: The number of epochs between pruning steps.
    :param use_config_initial_sparsity: Whether to use the initial sparsity from the config file.
                                        If True, model will be pruned to that sparsity in the first epoch,
                                        and it will be used as the initial sparsity for the gradual pruning.

    :return: The elapsed time for training, the logging list with the training information,
             and the time spent on pruning.
    """
    print('\nTraining model')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(initial_learning_rate))
    print('| Optimizer = SGD')

    best_acc, valid_accuracy, elapsed_time, pruning_elapsed_time = 0, 0, 0, 0
    layers_to_prune = get_layers(neural_network)
    pruning_method, goal_sparsity, block_size, block_criterion = pruning_params

    sparsity_info = get_sparsity(neural_network, layers_to_prune, pruning_params,
                                 # the pruning is either 0, or we cannot be certain it's been block pruned
                                 block_structure_check_enabled=False)
    print_sparsity_info(sparsity_info)
    initial_sparsity = sparsity_info['Total sparsity in pruned layers']
    if use_config_initial_sparsity and cf.initial_sparsity:
        print("Using initial sparsity from config file")
        initial_sparsity = cf.initial_sparsity  # because this is the sparsity from which the gradual pruning will start

    t0 = start_epoch
    if cf.t0:
        t0 = cf.t0
    print(f'| Gradual pruning starting params: n={n}, t0={t0}, delta_t={delta_t}, '
          f'initial_sparsity={initial_sparsity:.2f}, goal_sparsity={goal_sparsity:.2f}')

    last_epoch = start_epoch + num_epochs - 1
    logging_list = []

    check_params(last_epoch, n, delta_t, t0, initial_sparsity, goal_sparsity)

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):

        if (epoch - t0) % delta_t == 0:
            print("Applying gradual pruning...")
            current_goal_sparsity = calculate_current_goal_sparsity(epoch, n, t0, delta_t, initial_sparsity,
                                                                    goal_sparsity)
            print(f"Current goal sparsity: {current_goal_sparsity}")
            pruning_start_time = time.time()
            apply_pruning(neural_network, pruning_method, current_goal_sparsity, block_size, block_criterion)
            pruning_time = time.time() - pruning_start_time
            pruning_elapsed_time += pruning_time
            print("\nSparsity after pruning:")
            sparsity_info = get_sparsity(neural_network, layers_to_prune, (pruning_method, current_goal_sparsity,
                                         block_size, block_criterion),
                                         block_structure_check_enabled=(current_goal_sparsity > 0))
            print_sparsity_info(sparsity_info)

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
                             'valid_loss': valid_loss, 'epoch_time': epoch_time, 'epoch loss': epoch_train_loss,
                             'current_goal_sparsity': current_goal_sparsity})

    print(f"\nBest Model Accuracy: {best_acc:.2f}%")
    print(f"\nFinal Model Accuracy in Last Epoch: {valid_accuracy:.2f}%")

    return elapsed_time, logging_list, pruning_elapsed_time
