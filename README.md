# iterative-blocked-pruning-of-neural-networks
My diploma thesis on Iterative blocked pruning of neural networks.

The code builds upon the work https://github.com/meliketoy/wide-resnet.pytorch.

![Generate an illustrative picture of block pruning of matrices](https://github.com/user-attachments/assets/4b23d9d0-1e4b-4867-af5d-0c7069c114bb)
(A very insightful illustration by ChatGPT, prompt: Generate an illustrative picture of block pruning of matrices)

## Requirements
- Install appropriate [cuda](https://pytorch.org/get-started/locally/) version
- Download [Pytorch](https://pytorch.org)
- Clone the repository

Example:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## How to run
After you have cloned the repository, you can train the 28x10 Wide Residual Network network running the command below:
```bash
CUDA_VISIBLE_DEVICES=0 python code/main.py --filename_to_save 'my_network' --seed 31 
```
To apply pruning strategies, use appropriate arguments with your specific paths: 
```bash
CUDA_VISIBLE_DEVICES=0 python code/main.py --resume --resume_filepath './checkpoint/cifar100/my_network_lastepoch.t7' --filename_to_save "pruned_model" --pruning_method 'block' --block_criterion 'max' --sparsity 0.8 --seed 31
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

## Results
Logs and summary of results are included. One example of a well-performing trained 90% block-sparse model is stored under https://drive.google.com/drive/folders/1DXBIJpRNS6-2A5fva-whxalrU5c_5Oi5?usp=drive_link.
