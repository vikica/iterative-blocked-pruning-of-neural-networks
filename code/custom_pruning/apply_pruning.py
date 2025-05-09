import torch.nn
import torch.nn.utils.prune as prune
from custom_pruning.block_pruning import apply_block_pruning
from networks import Wide_ResNet
from custom_pruning.wide_resnet_layers_for_pruning import get_layers


def apply_pruning(neural_network: torch.nn.Module, pruning_method: str, sparsity: float, block_size: int,
                  block_criterion: str) -> None:
    """
    Prune the weights of the neural network.

    :param neural_network: the neural network to prune
    :param pruning_method: the pruning method to apply - global, block, block_iterative
    :param sparsity: the amount of sparsity of the weights after pruning (0.0 - 1.0)
    :param block_size: the size of the blocks for block pruning
    :param block_criterion: the criterion for selection of blocks to eliminate - L1, L2, max, min
    :return: None
    """

    # Check if network is of class 'networks.wide_resnet.Wide_ResNet'
    assert isinstance(neural_network, Wide_ResNet), "Pruning is only implemented for Wide_ResNet networks!"

    parameters_to_prune = get_layers(neural_network)

    if pruning_method == 'global_unstructured':
        print("Applying global unstructured pruning")
        # Globally prune tensors corresponding to all parameters in parameters by applying the specified pruning_method:
        prune.global_unstructured(
            [(layer, 'weight') for layer in parameters_to_prune],
            # prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm
            pruning_method=prune.L1Unstructured,
            amount=sparsity
        )
    elif pruning_method == 'block':
        print("Applying block pruning")
        apply_block_pruning(parameters_to_prune, amount=sparsity, block_size=block_size,
                            block_representative_function_name=block_criterion)
    elif pruning_method == 'block_iterative':
        print("Applying iterative block pruning")
        raise NotImplementedError
    else:
        raise ValueError("Invalid pruning method! Use 'global', 'block' or 'block_iterative'.")
