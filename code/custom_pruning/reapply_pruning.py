import torch
import torch.nn.utils.prune as prune


class ReapplyPruning(prune.BasePruningMethod):
    PRUNING_TYPE = 'global'

    def __init__(self, previous_pruning_mask):
        self.previous_pruning_mask = previous_pruning_mask

    def compute_mask(self, t, default_mask):
        return self.previous_pruning_mask


def reapply_pruning(module: torch.nn.Module, name: str, previous_pruning_mask: torch.Tensor) -> torch.nn.Module:
    """
    Reapply the pruning mask to the module.

    :param module: the module to which the pruning mask will be reapplied
    :param name: the name of the parameter in the module, e.g. 'weight'
    :param previous_pruning_mask: the pruning mask to be reapplied

    :return: the module with the pruning reapplied
    """
    ReapplyPruning.apply(module, name, previous_pruning_mask)
    return module
