import time
import datetime
import torch.optim as optim
import torch.nn.utils.prune as prune

import config as cf
from custom_pruning.utils import get_sparsity, print_sparsity_info
from custom_pruning.apply_pruning import apply_pruning
from custom_pruning.wide_resnet_layers_for_pruning import get_layers
from training_logic import train_one_epoch, test_network, save_checkpoint


def get_acdc_phase(epoch: int, warm_up_epochs: int, alternating_epochs: int, convergence_epochs: int,
                   total_epochs: int) -> str:
    """
    Get the current phase of the ACDC method based on the epoch number.

    :param epoch: The current epoch number.
    :param warm_up_epochs: The number of epochs for the warm-up phase.
    :param alternating_epochs: The number of epochs for alternating sparse-dense windows.
    :param convergence_epochs: The number of epochs for the final fine-tuning sparse phase.
    :param total_epochs: The total number of epochs for training.

    :return: The current phase of the ACDC method.
    """
    if epoch <= warm_up_epochs:
        return 'dense'
    if epoch > (total_epochs - convergence_epochs):
        return 'sparse'
    if ((epoch - 1 - warm_up_epochs) // alternating_epochs) % 2 == 0:
        return 'sparse'
    return 'dense'


def check_acdc_epochs(warm_up_epochs: int, alternating_epochs: int, convergence_epochs: int, total_epochs: int) -> None:
    """
    Check if the epochs for the ACDC method are set correctly.
    The ACDC method requires dense warmp_up, alternating the same number of sparse-dense windows, and a final
    fine-tuning sparse phase. These should add up to the total number of epochs.

    :param warm_up_epochs: The number of epochs for the warm-up phase.
    :param alternating_epochs: The number of epochs for alternating sparse-dense windows.
    :param convergence_epochs: The number of epochs for the final fine-tuning sparse phase.
    :param total_epochs: The total number of epochs for training.

    :raises ValueError: If the epochs do not add up correctly.
    """
    # Check if the parameters add up
    if (total_epochs - warm_up_epochs - convergence_epochs) % (alternating_epochs * 2) != 0:
        raise ValueError("The number of epochs for ACDC does not add up correctly! Please adjust the parameters.")


def acdc_train_network(neural_network, train_loader, test_loader, loss_criterion, start_epoch, num_epochs,
                       initial_learning_rate, file_path_for_checkpoint, device, pruning_params, warm_up_epochs=10,
                       alternating_epochs=20, convergence_epochs=30, alternating_lr=False):
    """
    Train the neural network using ACDC method.
    The training ends on a sparse phase, and we save the resulting sparse model, as well as the last dense model
    obtained at the end of the last dense phase.

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
    :param warm_up_epochs: The number of epochs for the warm-up phase.
    :param alternating_epochs: The number of epochs for alternating sparse-dense windows.
    :param convergence_epochs: The number of epochs for the final fine-tuning sparse phase.
    :param alternating_lr: Whether to use alternating learning rate (= a lower LR during dense phase).

    :return: The elapsed time for training, the logging list with the training information,
             and the time spent on pruning.
    """
    check_acdc_epochs(warm_up_epochs, alternating_epochs, convergence_epochs, num_epochs)

    print('\nTraining model using ACDC method')
    print('| Training Epochs = ' + str(num_epochs))
    print('| Initial Learning Rate = ' + str(initial_learning_rate))
    print('| Optimizer = SGD')

    best_sparse_acc, best_dense_acc, valid_accuracy, elapsed_time, pruning_elapsed_time = 0, 0, 0, 0, 0
    last_epoch = start_epoch + num_epochs - 1
    logging_list = []
    layers_to_prune = get_layers(neural_network)
    optimizer = optim.SGD(neural_network.parameters(), lr=cf.learning_rate(initial_learning_rate, start_epoch),
                          momentum=0.9, weight_decay=5e-4)
    last_phase = 'dense'

    start_time = time.time()

    for epoch in range(start_epoch, start_epoch + num_epochs):

        current_phase = get_acdc_phase(epoch, warm_up_epochs, alternating_epochs, convergence_epochs, num_epochs)
        print("\nCurrent phase: ", current_phase)

        current_learning_rate = cf.learning_rate(initial_learning_rate, epoch)
        if alternating_lr and current_phase == 'dense' and epoch > warm_up_epochs:
            # Set lower LR in dense phase
            current_learning_rate /= 3
        # Update the learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate

        if current_phase != last_phase:

            if current_phase == 'sparse':
                print("\nSparsity before pruning (after the end of dense phase):")
                sparsity_info = get_sparsity(neural_network, layers_to_prune,
                                             # There is supposed to be no pruning now
                                             ("none", None, None, None),
                                             block_structure_check_enabled=False)
                print_sparsity_info(sparsity_info)

                # Create new masks and prune the network
                pruning_start_time = time.time()
                apply_pruning(neural_network, *pruning_params)
                pruning_time = time.time() - pruning_start_time
                pruning_elapsed_time += pruning_time

                print("\nSparsity after pruning (=beginning of a sparse phase):")
                sparsity_info = get_sparsity(neural_network, layers_to_prune, pruning_params)
                print_sparsity_info(sparsity_info)

            elif current_phase == 'dense':

                # Switch to full masks
                print("Turning the pruning off")
                for module in layers_to_prune:
                    prune.remove(module, 'weight')

                print("\nSparsity at the beginning of a dense phase:")
                sparsity_info = get_sparsity(neural_network, layers_to_prune, pruning_params)
                print_sparsity_info(sparsity_info)

            else:
                raise ValueError(f"Invalid phase {current_phase}!")

        print(f"Training epoch: {epoch} / {last_epoch}")
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
            assert param_group['lr'] == current_learning_rate

        epoch_start_time = time.time()
        epoch_train_loss = train_one_epoch(neural_network, train_loader, loss_criterion, device, optimizer)
        # Validation step
        valid_accuracy, valid_loss = test_network(neural_network, test_loader, loss_criterion, device)
        epoch_time = time.time() - epoch_start_time
        print(f"| Validation Epoch #{epoch}\tLoss: {valid_loss:.4f} Acc@1: {valid_accuracy:.2f}%\t Epoch time: {epoch_time:.2f}s")
        elapsed_time = time.time() - start_time
        print(f'| Elapsed time: {datetime.timedelta(seconds=elapsed_time)}')

        # Save models
        if current_phase == 'dense' and valid_accuracy > best_dense_acc:
            print('| Saving the best dense model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time,
                            file_path_for_checkpoint + '_dense_bestepoch.t7')
            best_dense_acc = valid_accuracy
        if current_phase == 'sparse' and valid_accuracy > best_sparse_acc:
            print('| Saving the best sparse model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time,
                            file_path_for_checkpoint + '_sparse_bestepoch.t7')
            best_sparse_acc = valid_accuracy

        if epoch == num_epochs - convergence_epochs:
            print('| Saving the last dense model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time,
                            file_path_for_checkpoint + '_dense_lastepoch.t7')
        if epoch == last_epoch:
            print('| Saving the last sparse model...\t\t\tTop1 = %.2f%%' % valid_accuracy)
            save_checkpoint(neural_network, valid_accuracy, epoch, elapsed_time,
                            file_path_for_checkpoint+'_sparse_lastepoch.t7')

        logging_list.append({'epoch': epoch, 'learning_rate': current_learning_rate, 'valid_accuracy': valid_accuracy,
                             'valid_loss': valid_loss, 'epoch_time': epoch_time, 'epoch loss': epoch_train_loss,
                             'phase': current_phase, 'optimizer_momentum': 0.9})

        last_phase = current_phase

    print(f"\nBest Sparse Model Accuracy: {best_sparse_acc:.2f}%")
    print(f"Best Sparse Model Accuracy: {best_dense_acc:.2f}%")
    print(f"\nFinal Model Accuracy in Last Epoch: {valid_accuracy:.2f}%")

    return elapsed_time, logging_list, pruning_elapsed_time
