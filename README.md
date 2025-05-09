# iterative-blocked-pruning-of-neural-networks
My diploma thesis on Iterative blocked pruning of neural networks.

The code builds upon the work https://github.com/meliketoy/wide-resnet.pytorch.

## Requirements
- Install appropriate [cuda](https://pytorch.org/get-started/locally/) version
- Download [Pytorch](https://pytorch.org)
- Clone the repository

Example:
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## How to run
After you have cloned the repository, you can train the network running the command below.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --filename_to_save 'my_network' --seed 31 
```
To apply pruning strategies, use appropriate arguments with your specific paths: 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --resume --resume_filepath './checkpoint/cifar100/seed31/my_network_lastepoch.t7' --filename_to_save "seed31/pruned_model" --pruning_method 'block' --block_criterion 'max' --sparsity 0.8 --seed 31
```
(To see what other arguments are allowed, you can check the function ```parse_arguments()``` in ```main.py``` script)

Unit tests can be run like this from terminal: 
```bash
python unittests/tests.py
``` 
and 
```bash
python main_part/unittests/tests_custom_pruning.py
```
