import copy
import math
import unittest
from collections import OrderedDict
import torch.nn
from torch import nn
import sys
import os
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from custom_pruning.block_pruning import *
from custom_pruning.utils import *
from custom_pruning.reapply_pruning import reapply_pruning


block_criterion = "max"
device = "cpu"


class MyTestCase(unittest.TestCase):

    def test_find_block_representatives(self):
        matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
        block_size = 2
        block_representatives = find_block_representatives(matrix, block_size, "max")
        assert block_representatives == [6, 8, 14, 16]

    def test_find_block_representatives2(self):
        matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
        block_size = 4
        block_representatives = find_block_representatives(matrix, block_size, "max")
        assert block_representatives == [16]

    def test_find_threshold_linear2d(self):
        layer = torch.nn.Linear(4, 4)
        # set the layer weights
        weights = torch.tensor([[1, 1, 3, 4], [1, 1, 7, 8], [9, 10, 0.5, 0.4], [13, 14, 0.3, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 2
        sparsity_amount = 0.5
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        assert threshold == 1

    def test_block_pruning_linear2d(self):
        layer = torch.nn.Linear(4, 4)
        # set the layer weights
        weights = torch.tensor([[1, 1, 3, 4], [1, 1, 7, 8], [9, 10, 0.5, 0.4], [13, 14, 0.3, 0.2]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
            layer.bias = nn.Parameter(torch.tensor([1, 1, 1, 1], dtype=torch.float32))
        amount = 0.5
        block_size = 2
        apply_block_pruning(layer, amount, block_size, block_criterion)
        assert torch.equal(layer.weight, torch.tensor([[0, 0, 3, 4], [0, 0, 7, 8], [9, 10, 0, 0], [13, 14, 0, 0]], dtype=torch.float32))
        assert torch.equal(layer.bias, torch.tensor([1, 1, 1, 1], dtype=torch.float32))

    def test_find_threshold_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 2
        sparsity_amount = 0.5
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        self.assertEqual(threshold, 9)

    def test2_find_threshold_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 2
        sparsity_amount = 0.1
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        self.assertEqual(threshold, 3)

    def test3_find_threshold_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 2
        sparsity_amount = 0.25
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        self.assertEqual(threshold, 7)

    def test4_find_threshold_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 4
        sparsity_amount = 0.5
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        self.assertEqual(threshold, 9)

    def test5_find_threshold_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 4
        sparsity_amount = 0.8
        threshold = find_threshold([layer], sparsity_amount, block_size, "max")
        self.assertEqual(threshold, 13)

    def test_block_pruning_conv2d(self):
        layer = torch.nn.Conv2d(4, 4, 3)
        weights = torch.tensor([[[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]],
                                [[[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]], [[1, 1, 3], [4, 1, 1], [7, 8, 9]], [[9, 10, 0.5], [0.4, 13, 14], [0.3, 0.2, 0.1]]]], dtype=torch.float32)
        with torch.no_grad():
            layer.weight = nn.Parameter(weights)
        block_size = 4
        sparsity_amount = 0.8
        apply_block_pruning(layer, sparsity_amount, block_size, block_criterion)
        nonzero_weights = torch.sum(layer.weight != 0).item()
        self.assertEqual(nonzero_weights, 16)
        self.assertEqual(layer.weight[3, 3, 1, 1], 0)
        self.assertEqual(layer.weight[1, 3, 0, 1], 0)

    def test_block_pruning_unexpected_dimension_warning(self):
        layer = torch.nn.Conv2d(4, 4, 10)
        block_size = 4
        sparsity_amount = 0.8
        with self.assertWarns(Warning) as cm:
            apply_block_pruning(layer, sparsity_amount, block_size, block_criterion)
        # check the warning message
        self.assertEqual(str(cm.warning), "Unexpected weight dimension: the last 2 dimensions are neither (3,3) nor (1,1)!")

    def test_block_pruning_unexpected_dimension_error(self):
        layer = torch.nn.BatchNorm2d(4)
        block_size = 4
        sparsity_amount = 0.8
        with self.assertRaises(Exception) as cm:
            apply_block_pruning(layer, sparsity_amount, block_size, block_criterion)
        self.assertEqual(str(cm.exception), "Unexpected weight dimension dim=1! The weight tensor is expected to have 2 or 4 dimensions.")

    def test_block_pruning_model_with_multiple_layers(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 6, 3)),
            ('conv2', nn.Conv2d(6, 16, 3)),
            ('lin1', nn.Linear(16 * 5 * 5, 120)),
            ('lin2', nn.Linear(120, 84)),
            ('lin3', nn.Linear(84, 10))
        ]))
        parameters_to_prune = [model.conv1, model.conv2, model.lin1, model.lin2, model.lin3]
        block_size = 2
        sparsity_amount = 0.5
        apply_block_pruning(parameters_to_prune, sparsity_amount, block_size, block_criterion)
        sparsity_info = get_sparsity(model, parameters_to_prune,
                                     ("block", sparsity_amount, block_size, block_criterion))
        self.assertEqual(sparsity_info['Global sparsity'], 0.5)

    def test2_block_pruning_model_with_multiple_layers(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 6, 3)),
            ('conv2', nn.Conv2d(6, 16, 3)),
            ('lin1', nn.Linear(16 * 5 * 5, 120)),
            ('lin2', nn.Linear(120, 84)),
            ('lin3', nn.Linear(84, 10))
        ]))
        parameters_to_prune = [model.conv1, model.conv2, model.lin1, model.lin2, model.lin3]
        block_size = 2
        sparsity_amount = 0.8
        apply_block_pruning(parameters_to_prune, sparsity_amount, block_size, block_criterion)
        sparsity_info = get_sparsity(model, parameters_to_prune,
                                     ("block", sparsity_amount, block_size, block_criterion))
        self.assertEqual(sparsity_info['Global sparsity'], 0.8)

    def test3_block_pruning_model_with_multiple_layers(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(2, 2, 3)),
            ('lin1', nn.Linear(2, 2))
        ]))
        parameters_to_prune = [model.conv1, model.lin1]
        block_size = 2
        sparsity_amount = 0.5
        apply_block_pruning(parameters_to_prune, sparsity_amount, block_size, block_criterion)
        sparsity_info = get_sparsity(model, parameters_to_prune,
                                     ("block", sparsity_amount, block_size, block_criterion))
        self.assertEqual(sparsity_info['Global sparsity'], 0.5)

    def test_block_pruning_with_block_size_1(self):
        model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(4, 6, 3)),
            ('conv2', nn.Conv2d(6, 16, 3)),
            ('lin1', nn.Linear(16 * 5 * 5, 120)),
            ('lin2', nn.Linear(120, 84)),
            ('lin3', nn.Linear(84, 10))
        ]))
        model2 = copy.deepcopy(model)

        parameters_to_prune = [model.conv1, model.conv2, model.lin1, model.lin2, model.lin3]
        block_size = 1
        sparsity_amount = 0.8
        apply_block_pruning(parameters_to_prune, sparsity_amount, block_size, block_criterion)
        sparsity_info = get_sparsity(model, parameters_to_prune,
                                     ("block", sparsity_amount, block_size, block_criterion))
        self.assertEqual(sparsity_info['Global sparsity'], 0.8)

        sparsity_info2 = get_sparsity(model2, parameters_to_prune,
                                      ("none", 0, 0, "none"))
        self.assertEqual(sparsity_info2['Global sparsity'], 0.0)
        parameters_to_prune = ((model2.conv1, 'weight'),
                               (model2.conv2, 'weight'),
                               (model2.lin1, 'weight'),
                               (model2.lin2, 'weight'),
                               (model2.lin3, 'weight'))
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=sparsity_amount,
        )
        sparsity_info2 = get_sparsity(model2, parameters_to_prune,
                                      ("global_unstructured", sparsity_amount, 0, "none"))
        self.assertEqual(sparsity_info2['Global sparsity'], 0.8)

        assert torch.equal(model.conv1.weight, model2.conv1.weight)
        assert torch.equal(model.conv2.weight, model2.conv2.weight)
        assert torch.equal(model.lin1.weight, model2.lin1.weight)
        assert torch.equal(model.lin2.weight, model2.lin2.weight)
        assert torch.equal(model.lin3.weight, model2.lin3.weight)

    def test_block_representative_function_min(self):
        matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
        block_size = 2
        block_representatives = find_block_representatives(matrix, block_size, "min")
        self.assertEqual(block_representatives, [1, 3, 9, 11])

    def test_block_representative_function_L1(self):
        matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
        block_size = 2
        block_representatives = find_block_representatives(matrix, block_size, "L1")
        self.assertEqual(block_representatives, [14, 22, 46, 11+12+15+16])

    def test_block_representative_function_L2(self):
        matrix = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=torch.float32)
        block_size = 2
        block_representatives = find_block_representatives(matrix, block_size, "L2")
        expected_values = [
            math.sqrt(1 + 4 + 25 + 36),
            math.sqrt(9 + 16 + 49 + 64),
            math.sqrt(9 ** 2 + 10 ** 2 + 13 ** 2 + 14 ** 2),
            math.sqrt(11 ** 2 + 12 ** 2 + 15 ** 2 + 16 ** 2)
        ]
        for br, ev in zip(block_representatives, expected_values):
            self.assertAlmostEqual(br, ev, places=4)

    def test_block_pruning_sparsity_1(self):
        # Create a 2D convolutional layer with kernel size 1x1
        conv_layer = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        # Initialize the weights with basic numbers (e.g., sequential integers)
        basic_numbers = torch.arange(1, 257).view(16, 16, 1, 1).float()  # Create a tensor of size (16, 16, 1, 1)
        conv_layer.weight.data = basic_numbers  # Assign the tensor as weights
        conv_layer.bias.data.fill_(0)  # Initialize bias to zero

        class LeNet(nn.Module):
            def __init__(self):
                super(LeNet, self).__init__()
                self.conv1 = conv_layer

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, int(x.nelement() / x.shape[0]))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = LeNet().to(device=device)
        parameters_to_prune = [model.conv1]
        apply_block_pruning(parameters_to_prune, amount=1, block_size=8, block_representative_function_name="max")
        assert get_sparsity(model, parameters_to_prune, ("block", 1, 8, "max"))['Global sparsity'] == 1

    def test_iterative_block_pruning(self):

        # Create a 2D convolutional layer with kernel size 1x1
        conv_layer = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        # Initialize the weights with basic numbers (e.g., sequential integers)
        basic_numbers = torch.arange(1, 257).view(16, 16, 1, 1).float()  # Create a tensor of size (16, 16, 1, 1)
        conv_layer.weight.data = basic_numbers  # Assign the tensor as weights
        conv_layer.bias.data.fill_(0)  # Initialize bias to zero

        class LeNet(nn.Module):
            def __init__(self):
                super(LeNet, self).__init__()
                self.conv1 = conv_layer

            def forward(self, x):
                x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
                x = F.max_pool2d(F.relu(self.conv2(x)), 2)
                x = x.view(-1, int(x.nelement() / x.shape[0]))
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        model = LeNet().to(device=device)
        parameters_to_prune = [model.conv1]
        apply_block_pruning(parameters_to_prune, amount=0.25, block_size=8, block_representative_function_name="max")
        assert get_sparsity(model, parameters_to_prune, ("block", 0.25, 8, "max"))['Global sparsity'] == 0.25
        apply_block_pruning(parameters_to_prune, amount=0.34, block_size=8, block_representative_function_name="max")
        assert get_sparsity(model, parameters_to_prune, ("block", 0.5, 8, "max"))['Global sparsity'] == 0.5

    def test_threshold(self):
        lst = [1.0, 128.0, 248.0, 256.0]
        assert determine_threshold(lst, 0.25) == 1
        assert determine_threshold(lst, 0.2501) == 128
        assert determine_threshold(lst, 0.5) == 128
        assert determine_threshold(lst, 0.501) == 248
        assert determine_threshold(lst, 0.75) == 248
        assert determine_threshold(lst, 1) == 256

    def test_get_weights_mask_existing_mask(self):
        conv_layer = nn.Conv2d(16, 16, 1)
        apply_block_pruning(conv_layer, 0.5, 8, "max")
        mask = get_weight_mask(conv_layer)
        assert mask is not None
        # Check the return type
        assert isinstance(mask, torch.Tensor)

    def test_get_weights_mask_nonexisting_mask(self):
        conv_layer = nn.Conv2d(16, 16, 1)
        # Value Error is supposed to be thrown
        with self.assertRaises(ValueError):
            get_weight_mask(conv_layer)

    def test_get_weights_mask_nonexisting_mask_after_removed_pruning(self):
        conv_layer = nn.Conv2d(16, 16, 1)
        prune.global_unstructured([(conv_layer, 'weight')], pruning_method=prune.L1Unstructured, amount=0.8)
        prune.remove(conv_layer, 'weight')
        with self.assertRaises(ValueError):
            get_weight_mask(conv_layer)

    def test_reapply_pruning(self):
        conv_layer = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, stride=1, padding=0)
        basic_numbers = torch.arange(1, 257).view(16, 16, 1, 1).float()
        conv_layer.weight.data = basic_numbers
        conv_layer.bias.data.fill_(0)

        # Apply block pruning
        parameters_to_prune = conv_layer
        apply_block_pruning(parameters_to_prune, 0.8, 8, "max")
        assert list(conv_layer.named_buffers()) != []
        pruning_mask = get_weight_mask(conv_layer)

        # Remove the pruning
        prune.remove(conv_layer, 'weight')

        # Reapply the pruning
        reapply_pruning(conv_layer, 'weight', pruning_mask)

        assert all(torch.equal(a, b) for a, b in zip(get_weight_mask(conv_layer), pruning_mask))

        # Check whether get_weight_mask(conv_layer) contains at least one zero
        assert torch.sum(get_weight_mask(conv_layer) == 0) > 0


if __name__ == '__main__':
    unittest.main()
