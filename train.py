import os
import sys

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

import torch
import torchvision
import torchvision.transforms as transforms

import torch.optim as optim
from torch.autograd import Variable

from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from PIL import Image

from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim

is_cuda = torch.cuda.is_available()

os.environ["RANK"] = "0"

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destropy_process_group

def main():
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    if not len(args.gpu) > torch.cuda.device_count():
        ngpus_per_node = len(args.gpu)
    else:
        print("We will use all available GPUs")
        ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)

    model = DenseFuse_net(input_nc=args.CHANNELS, output_nc=args.CHANNELS)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    epoch = 0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.resume:
        if args.gpu is None:
            checkpoint = torch.load(args.resume)
        else:
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.resume, map_location=loc)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else:
        # print("Training from scratch")
        pass
    
    img_path_file = args.dataset

    assert args.CHANNELS == 1 or 3, "Input channels should be either 1 or 3"
    if args.CHANNELS == 1:
        custom_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                            transforms.Resize((args.HEIGHT,args.WIDTH)),
                                            transforms.ToTensor()])
    elif args.CHANNELS == 3:
        custom_transform = transforms.Compose([transforms.Resize((args.HEIGHT,args.WIDTH)),
                                            transforms.ToTensor()])

    trainloader = DataLoader(MyTrainDataset(img_path_file, custom_transform=custom_transform), batch_size=args.batch_size, shuffle=False, num_workers=4)

    for ep in range(epoch, args.epochs):

        pbar = tqdm(trainloader)

        for inputs in pbar:
        # for inputs in trainloader:
            
            if args.gpu is not None:
                inputs = inputs.cuda(args.gpu, non_blocking=True)

            optimizer.zero_grad()
            
            en = model.encoder(inputs)
            predicts = model.decoder(en)

            loss = compute_loss(predicts, inputs, args.ssim_weight, w_idx=2)
            loss.backward()
            optimizer.step()

        if (ep + 1) % args.save_per_epoch == 0:
            # Save model
            torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss
                    }, args.save_model_dir + 'ckpt_{}.pt'.format(ep))

    print('Finished training')

class MyTrainDataset(Dataset):
    def __init__(self, img_path_file, custom_transform=transforms.Compose([transforms.ToTensor()])):
        f = open(img_path_file, 'r')
        img_list = f.read().splitlines()
        f.close()

        self.img_list = img_list
        self.transform = custom_transform
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index]).convert('RGB')

        if self.transform:
            img = self.transform(img)
        
        return img

    def __len__(self):

        return len(self.img_list)

def compute_loss(predicts, targets, weights, w_idx=0):

    # targets = Variable(targets.data.clone(), requires_grad=False)
    mse = nn.MSELoss()
    ssim = pytorch_msssim.SSIM()

    loss_mse = mse(predicts, targets)
    loss_ssim = 1 - ssim(predicts, targets)

    loss = loss_mse + weights[w_idx] * loss_ssim
    return loss

if __name__ == "__main__":
    main()


