#!/att/nobackup/jframe/anaconda3/bin/python3
import multiprocessing
import torch
num_cores = multiprocessing.cpu_count()
print(f"There are {num_cores} cpu cores available")
use_cuda = torch.cuda.is_available()
print(use_cuda)
DEVICE = torch.device("cuda:0" if use_cuda else "cpu")
print('using device', DEVICE)
