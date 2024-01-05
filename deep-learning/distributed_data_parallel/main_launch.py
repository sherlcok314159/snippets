import tempfile
from pathlib import Path
from dataclasses import dataclass

import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, DistributedSampler

from utils import *

def plot_results(data, *labels):
    plt.plot(data)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.savefig('./results/%s.png'%labels[1])
    plt.cla()

@dataclass
class MNISTConfig:
    lr: float = 2e-4
    epochs: int = 5
    num_workers: int = 8
    per_gpu_train_bs: int = 256
    per_gpu_eval_bs: int = 512
    syncBN: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    backend: str = 'nccl'
    init_method: str = 'tcp://127.0.0.1:23456'
    devices: str = '0, 1'
    seed: int = 0

class MNISTNet(nn.Module):
   def __init__(self):
      super().__init__()
      self.layer = nn.Sequential(
                   nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3), 
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=2),
                   nn.Conv2d(32, 64, 2),
                   nn.ReLU(),
                   nn.MaxPool2d(2, 2),
                   nn.Flatten(),
                   nn.Linear(64 * 6 * 6, 10),
                )

   def forward(self, x):
      x = self.layer(x)
      return x
    
def prepare_data(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, ),(0.5, ))
    ])
    train_dataset = datasets.MNIST(root='./data',
                                   transform=transform,
                                   train=True,
                                   download=True)

    eval_dataset = datasets.MNIST(root='./data',
                                  transform=transform,
                                  train=False)

    train_sampler = DistributedSampler(train_dataset, seed=args.seed)
    train_dataloader = DataLoader(train_dataset,
                                  pin_memory=True,
                                  shuffle=(train_sampler is None),
                                  batch_size=args.per_gpu_train_bs,
                                  num_workers=args.num_workers,
                                  sampler=train_sampler)

    eval_sampler = DistributedSampler(eval_dataset, seed=args.seed)
    eval_dataloader = DataLoader(eval_dataset,
                                 pin_memory=True,
                                 batch_size=args.per_gpu_eval_bs,
                                 num_workers=args.num_workers,
                                 sampler=eval_sampler)

    return train_dataloader, eval_dataloader

def train(
    rank,
    world_size,
    model, 
    dataloader,
    optimizer,
):
    model.train()
    mean_loss = 0
    loss_cache = []
    # set up tqdm for the main process
    if is_main_process(rank):
        dataloader = tqdm(dataloader, ncols=80)

    for step, (inputs, labels) in enumerate(dataloader):
        inputs, labels = map(lambda x: x.cuda(), (inputs, labels))
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels) 
        loss.backward() 
        # here we just need to record the mean loss
        # DDP automatically averages the accumulated gradients
        # [g /= world_size] after we call loss.backward()
        loss = reduce_value(loss, world_size)
        mean_loss = (step * mean_loss + loss.item()) / (step + 1)
        loss_cache.append(mean_loss)

        optimizer.step() 
        optimizer.zero_grad()
        model.zero_grad()

    return loss_cache

@torch.no_grad()
def eval(
    rank,
    world_size,
    model, 
    dataloader,
):
    model.eval()
    # BoolTensor is not supported
    results = torch.tensor([]).cuda()
    # set up tqdm for the main process
    if is_main_process(rank):
        dataloader = tqdm(dataloader, ncols=80)

    for step, (inputs, labels) in enumerate(dataloader):
        inputs, labels = map(lambda x: x.cuda(), (inputs, labels))
        outputs = model(inputs)
        res = (outputs.argmax(-1) == labels)
        results = torch.cat([results, res], dim=0)

    results = sync_across_gpus(results, world_size)
    mean_acc = (results.sum() / len(results)).item()
    return mean_acc

def main():
    # the arguments
    args = MNISTConfig()

    dist_setup_launch(args)
    rank = args.rank
    world_size = args.world_size
    local_rank = args.local_rank

    # you may wonder why not multiply lr by sqrt(world_size) to keep the same variance.
    # you can check: https://arxiv.org/abs/1706.02677
    # and https://github.com/Lightning-AI/lightning/discussions/3706
    lr = args.lr * world_size
    
    # wrap the model by DDP
    model = MNISTNet()

    if args.syncBN:
        nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = model.cuda()
    model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, 
                                                device_ids=[rank])
    optimizer = Adam(model.parameters(), lr)

    train_dataloader, eval_dataloader = prepare_data(args)

    loss_caches, acc_caches, best_acc = [], [], 0
    # disappear when the program is done
    save_path = Path(tempfile.gettempdir(), 'pytorch_model.bin')

    for epoch in range(args.epochs):
        # make shuffled data different across epochs
        train_dataloader.sampler.set_epoch(epoch)
        loss_cache = train(rank, world_size, model, train_dataloader, optimizer)
        mean_acc = eval(rank, world_size, model, eval_dataloader)
        # save the checkpoint if acc is better
        if best_acc < mean_acc:
            best_acc = mean_acc
            save_checkpoint(rank, model, save_path)
        
        if is_main_process(rank):
            loss_caches += loss_cache
            acc_caches.append(mean_acc)
            print('Mean Acc in Epoch %d: %.3f' %(epoch, mean_acc))
        
    load_checkpoint(rank, model, save_path)
    final_acc = eval(rank, world_size, model, eval_dataloader)

    if is_main_process(rank):
        plot_results(loss_caches, 'Steps', 'Loss')
        plot_results(acc_caches, 'Epochs', 'Acc')
        print('The best acc is %.3f' %final_acc)
    
    cleanup()
    
if __name__ == '__main__':
    main()