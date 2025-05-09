import warnings
import torch


def calculate_sparsity(module: torch.nn.Module) -> (float, int, int):
    """
    Calculate the sparsity rate of weights for the given module.

    :param module: the module for which the sparsity rate will be calculated

    :return: the sparsity rate of weights for the given module, the number of non-zero weights,
             and the total number of weights
    """
    nonzero_count = torch.sum(module.weight != 0).item()
    total = module.weight.nelement()
    sparsity_rate = (total - nonzero_count) / total
    return sparsity_rate, nonzero_count, total


def check_desired_vs_real_sparsity(desired_sparsity: float, real_sparsity: float, tolerance_percentage: int = 1) -> None:
    """
    Check if the desired sparsity and the real sparsity differ by more than the tolerance percentage.
    If they do, print a warning.

    :param desired_sparsity: the desired sparsity rate
    :param real_sparsity: the real sparsity rate (in pruned layers)
    :param tolerance_percentage: the tolerance percentage for the difference between desired and real sparsity
    """
    if abs(desired_sparsity - real_sparsity) > 0.01 * tolerance_percentage:
        warnings.warn(f"Warning: Desired sparsity ({desired_sparsity}) and real sparsity (in pruned layers)"
                      f"({real_sparsity}) differ by more than {tolerance_percentage} percent!")


def calculate_number_of_zero_blocks(matrix_2d, block_size):
    """
    Calculate the number of zero blocks in the given 2D matrix.

    :param matrix_2d: the 2D matrix
    :param block_size: the size of the blocks

    :return: the number of zero blocks
    """
    number_of_zero_blocks = 0
    for i in range(0, matrix_2d.size(0), block_size):
        for j in range(0, matrix_2d.size(1), block_size):
            block = matrix_2d[i:i + block_size, j:j + block_size]
            # Check if all elements in the block are zero
            block_is_zero = (torch.sum(block != 0) == 0)
            if block_is_zero:
                number_of_zero_blocks += 1
    return number_of_zero_blocks


def analyze_blocks(module, block_size):
    """
    We need this function to check if the module fulfills the blocked pruning structure.
    The module is expected to have 2 or 4 dimensions.
    If the module has 4 dimensions, we permute the last 2 dimensions to pretend it is 2D.

    :param module: the module to check
    :param block_size: the size of the blocks

    :return: the number of zero blocks, the number of elements pruned in blocks,
             and the sparsity induced by block pruning
    """
    dim = len(module.weight.shape)
    number_of_zero_blocks = 0
    if dim == 2:
        number_of_zero_blocks = calculate_number_of_zero_blocks(module.weight, block_size)
    elif dim == 4:
        # we'll pretend the 4D tensor is 2D for now (the last 2 dimensions (=kernel) "do not exist")
        weights_permuted = torch.permute(module.weight, (2, 3, 0, 1))
        if (weights_permuted.size(0) != 3 or weights_permuted.size(1) != 3) and (
                weights_permuted.size(0) != 1 or weights_permuted.size(1) != 1):
            # In WideResnet, Conv2D layers have 3x3 or 1x1 kernel size
            warnings.warn("Unexpected weight dimension: the last 2 dimensions are neither (3,3) nor (1,1)!")

        # the blocks are in the last 2 dimensions:
        for i in range(0, weights_permuted.size(0)):
            for j in range(0, weights_permuted.size(1)):
                matrix = weights_permuted[i, j]
                number_of_zero_blocks += calculate_number_of_zero_blocks(matrix, block_size)
    else:
        raise Exception(f"Unexpected weight dimension dim={dim}! The weight tensor is expected to have 2 or 4 "
                        f"dimensions.")
    number_of_pruned_elements = number_of_zero_blocks * block_size * block_size
    total_number_of_elements = module.weight.nelement()
    block_sparsity = number_of_pruned_elements / total_number_of_elements

    return number_of_zero_blocks, number_of_pruned_elements, block_sparsity


def check_block_pruning_structure(total_number_of_zero_blocks, block_size, desired_sparsity,
                                  pruned_layers_elements_count):
    """
    Check if the block pruning structure is correct. The block pruning structure is correct if the number of zero blocks
    is greater than 0 and the sparsity induced by block pruning is (ca.) equal to the desired sparsity.

    :param total_number_of_zero_blocks: the total number of zero blocks in the model
    :param block_size: the size of the blocks
    :param desired_sparsity: the desired sparsity rate
    :param pruned_layers_elements_count: the total number of elements in pruned layers

    :raises ValueError: if no pruned blocks are found
    """
    if total_number_of_zero_blocks == 0:
        raise ValueError("Block structure check failed: The block pruning is not applied: no pruned blocks found in "
                         "the model!")
    total_blocked_pruned_elements = total_number_of_zero_blocks * block_size * block_size
    blocked_pruning_rate_in_pruned_layers = total_blocked_pruned_elements / pruned_layers_elements_count
    desired_vs_real_block_sparsity_dif = abs(desired_sparsity - blocked_pruning_rate_in_pruned_layers)
    if desired_vs_real_block_sparsity_dif == 0:
        print(
            f"Block structure check passed: Desired sparsity {desired_sparsity} and sparsity induced by block pruning "
            f"{blocked_pruning_rate_in_pruned_layers} are equal")
    else:
        warnings.warn(
            f"Block structure check warning: Desired sparsity {desired_sparsity} and sparsity induced by block pruning "
            f"{blocked_pruning_rate_in_pruned_layers} differ by "
            f"{abs(blocked_pruning_rate_in_pruned_layers - desired_sparsity)}")
        if desired_vs_real_block_sparsity_dif > 0.01:
            warnings.warn(
                f"Block structure check failed: Desired sparsity {desired_sparsity} and sparsity induced by block "
                f"pruning {blocked_pruning_rate_in_pruned_layers} differ by more than 1%!!!")


def get_sparsity(model, pruned_layers, pruning_params, block_structure_check_enabled=True):
    """
    Calculate the sparsity rate of weights for all layers in the model, as well as the global sparsity rate.
    The sparsity rate is calculated as the number of zero weights divided by the total number of weights.
    If block pruning is applied and block_structure_check_enabled, the function also checks the block pruning structure.
    Caution: The block pruning structure check may take a long time to be evaluated.

    :param model: the neural network model
    :param pruned_layers: the list of pruned layers
    :param pruning_params: the tuple of pruning parameters:
                           (pruning_method, desired_sparsity, block_size, block_criterion),
                           desired_sparsity can be None
    :param block_structure_check_enabled: whether to check the block pruning structure (we may want to disable it for
                                          ACDC or other cases) when pruning_method is 'block' or 'block_iterative'.
                                          Default is True.

    :return: a dictionary containing the sparsity rate of weights for all layers in the model, as well as the sparsity
             rate of weights in pruned layers and the global sparsity rate
    """
    pruning_method, desired_sparsity, block_size, _ = pruning_params
    total_nonzero, total_elements, pruned_layers_nonzero_elements_count, pruned_layers_elements_count = 0, 0, 0, 0
    total_number_of_zero_blocks = 0
    sparsity_info = {}

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            assert isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d))

            sparsity, nonzero, total = calculate_sparsity(module)
            sparsity_info[name + '.weight'] = sparsity
            total_nonzero += nonzero
            total_elements += total

            if module in pruned_layers:
                pruned_layers_nonzero_elements_count += nonzero
                pruned_layers_elements_count += total

                if block_structure_check_enabled and (pruning_method == 'block' or pruning_method == 'block_iterative'):
                    number_of_zero_blocks, _, _ = analyze_blocks(module, block_size)
                    total_number_of_zero_blocks += number_of_zero_blocks

    if block_structure_check_enabled and (pruning_method == 'block' or pruning_method == 'block_iterative'):
        check_block_pruning_structure(total_number_of_zero_blocks, block_size, desired_sparsity,
                                      pruned_layers_elements_count)

    if pruned_layers_elements_count > 0:
        pruned_layers_sparsity = (pruned_layers_elements_count - pruned_layers_nonzero_elements_count) / pruned_layers_elements_count
        sparsity_info['Total sparsity in pruned layers'] = pruned_layers_sparsity
        if desired_sparsity is not None:
            check_desired_vs_real_sparsity(desired_sparsity, pruned_layers_sparsity)

    global_sparsity = (total_elements - total_nonzero) / total_elements
    sparsity_info['Global sparsity'] = global_sparsity

    return sparsity_info


def print_sparsity_info(sparsity_info: dict) -> None:
    print("Printing sparsity of the network:")
    for layer, sparsity in sparsity_info.items():
        print(f"{layer}: {sparsity*100:.2f}%")
    print()


def get_weight_mask(module: torch.nn.Module) -> torch.Tensor:
    """
    Get the pruning mask for weights of the module.
    This function is supposed to be called on module that has been (and still is) pruned.

    :param module: the pruned module for which the pruning mask will be returned

    :return: the pruning mask for weights of the module

    :raises ValueError: if the module does not contain a 'weight_mask' buffer
    """
    for name, buffer in module.named_buffers():
        if name == "weight_mask":
            return buffer

    raise ValueError(f"Module {module} does not contain a 'weight_mask'. Ensure pruning has been applied.")
