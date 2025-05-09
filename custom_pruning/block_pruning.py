import warnings
import torch
import torch.nn.utils.prune as prune
import numpy as np


# mapping of function names to actual functions
block_representative_funcs = {
    "max": lambda block: torch.max(torch.abs(block)).item(),
    "min": lambda block: torch.min(torch.abs(block)).item(),
    "L1": lambda block: torch.norm(block, p=1).item(),
    "L2": lambda block: torch.norm(block, p=2).item(),
}


class BlockPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, block_size, dim, threshold, block_representative_function_name):
        self.block_size = block_size
        self.dim = dim  # dimension of the weight tensor
        self.threshold = threshold
        self.block_representative_function_name = block_representative_function_name

    def compute_mask(self, t, default_mask):
        """
        Compute the mask for the weight tensor t. The blocks of weights that are below the threshold will be pruned.
        :param t: the weight tensor
        :param default_mask:
        :return: the pruning mask
        """
        mask = default_mask.clone()
        # custom logic - zero out blocks of the tensor
        if self.dim == 2:
            for i in range(0, t.size(0), self.block_size):
                for j in range(0, t.size(1), self.block_size):
                    block = t[i:i + self.block_size, j:j + self.block_size]
                    block_representative = block_representative_funcs[self.block_representative_function_name](block)
                    if block_representative <= self.threshold:
                        mask[i:i + self.block_size, j:j + self.block_size] = 0
        elif self.dim == 4:
            mask_permuted = torch.permute(mask, (2, 3, 0, 1))
            t_permuted = torch.permute(t, (2, 3, 0, 1))
            for i in range(0, t_permuted.size(0)):
                for j in range(0, t_permuted.size(1)):
                    matrix = t_permuted[i, j]
                    for k in range(0, matrix.size(0), self.block_size):
                        for m in range(0, matrix.size(1), self.block_size):
                            block = matrix[k:k + self.block_size, m:m + self.block_size]
                            block_representative = block_representative_funcs[self.block_representative_function_name](block)
                            if block_representative <= self.threshold:
                                mask_permuted[i, j, k:k + self.block_size, m:m + self.block_size] = 0
            mask = torch.permute(mask_permuted, (2, 3, 0, 1))
        else:
            raise Exception(f"Unexpected weight dimension dim={self.dim}! The weight tensor is expected to have 2 or 4 "
                            f"dimensions.")
        return mask


def find_block_representatives(matrix, block_size, block_representative_function_name):
    """
    Find the representatives of the blocks in the matrix.
    :param matrix: a 2D tensor
    :param block_size: the size of the blocks for pruning
    :param block_representative_function_name: the method to find the block representative: "max", "min", "L1", "L2"
    :return: a list of block representatives
    """
    block_representative_function = block_representative_funcs[block_representative_function_name]
    block_representatives = []
    for i in range(0, matrix.size(0), block_size):
        for j in range(0, matrix.size(1), block_size):
            block = matrix[i:i + block_size, j:j + block_size]
            block_representative = block_representative_function(block)
            block_representatives.append(block_representative)
    return block_representatives


def determine_threshold(block_representatives, sparsity_amount):
    return np.percentile(block_representatives, sparsity_amount * 100, method="inverted_cdf")


def find_threshold(parameters_to_prune, sparsity_amount, block_size, block_representative_function_name):
    """
    Find the threshold for block pruning based on the sparsity_amount.
    :param parameters_to_prune: a list or tuple containing modules to prune
    :param sparsity_amount: the desired sparsity amount (0.0 - 1.0)
    :param block_size: the size of the blocks for pruning
    :param block_representative_function_name: the method to find the block representative: "max", "min", "L1", "L2"
    :return: float, the threshold for block pruning
    """
    if sparsity_amount == 0:
        return 0
    if sparsity_amount == 1:
        return float("inf")
    block_representatives = []
    for module in parameters_to_prune:
        t = module.weight
        dim = len(t.shape)
        if dim == 2:
            matrix_block_representatives = find_block_representatives(t, block_size, block_representative_function_name)
            block_representatives += matrix_block_representatives  # list concatenation
        elif dim == 4:
            # we'll pretend the 4D tensor is 2D for now (the last 2 dimensions (=kernel) "do not exist")
            weights_permuted = torch.permute(module.weight, (2, 3, 0, 1))
            if (weights_permuted.size(0) != 3 or weights_permuted.size(1) != 3) and (
                    weights_permuted.size(0) != 1 or weights_permuted.size(1) != 1):
                warnings.warn("Unexpected kernel size: the last 2 dimensions are neither (3,3) nor (1,1)!")
                # Because in WideResnet, Conv2D layers have 3x3 or 1x1 kernel size.
            # the blocks are in the last 2 dimensions - we'll make blocks for (usually) 9 separate matrices:
            for i in range(0, weights_permuted.size(0)):
                for j in range(0, weights_permuted.size(1)):
                    matrix = weights_permuted[i, j]
                    matrix_block_representatives = find_block_representatives(matrix, block_size,
                                                                              block_representative_function_name)
                    block_representatives += matrix_block_representatives  # list concatenation
        else:
            raise Exception(f"Unexpected weight dimension dim={dim}! The weight tensor is expected to have 2 or 4 "
                            f"dimensions.")

    threshold = determine_threshold(block_representatives, sparsity_amount)
    return threshold


def apply_block_pruning(parameters_to_prune, amount, block_size, block_representative_function_name):
    """
    Apply block pruning to the weights of the modules in parameters_to_prune.
    The resulting sparsity amount will not always be exactly the same as the desired amount.
    :param parameters_to_prune: a module or a list/tuple of modules to prune
    :param amount: the sparsity amount (0.0 - 1.0)
    :param block_size: the size of the blocks for pruning
    :param block_representative_function_name: the method to find the block representative: "max", "min", "L1", "L2"
    :return: the modules with pruned weights
    """
    if type(parameters_to_prune) not in (list, tuple):
        parameters_to_prune = [parameters_to_prune]
    threshold = find_threshold(parameters_to_prune, amount, block_size, block_representative_function_name)
    print("Threshold for pruning calculated:", threshold)
    for module in parameters_to_prune:
        BlockPruning.apply(module, 'weight', block_size=block_size, dim=len(module.weight.shape),
                           threshold=threshold, block_representative_function_name=block_representative_function_name)
    return parameters_to_prune

