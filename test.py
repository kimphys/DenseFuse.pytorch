import os
import sys

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from PIL import Image

from net import DenseFuse_net
from args_fusion import args
import pytorch_msssim

is_cuda = torch.cuda.is_available()

class MyTestDataset(Dataset):
    def __init__(self, img_path_file, ir_path_file):
        f_img = open(img_path_file, 'r')
        f_ir = open(ir_path_file, 'r')
        img_list = f_img.read().splitlines()
        ir_list = f_ir.read().splitlines()
        f_img.close()
        f_ir.close()

        self.img_list = img_list
        self.ir_list = ir_list
        self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                             transforms.Resize((args.HEIGHT,args.WIDTH)),
                                             transforms.ToTensor()])
    
    def __getitem__(self, index):
        img = Image.open(self.img_list[index])
        ir = Image.open(self.ir_list[index])

        if self.transform:
            img = self.transform(img)
            ir = self.transform(ir)
        
        return img, ir

    def __len__(self):

        return len(self.img_list)

if __name__ == '__main__':

    model = DenseFuse_net()

    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    strategy_type = args.strategy_type

    img_path_file = args.test_img
    ir_path_file = args.test_ir
    testloader = DataLoader(MyTestDataset(img_path_file, ir_path_file), batch_size=1, shuffle=False, num_workers=1)

    if is_cuda:
        model.cuda()

    for i, (img, ir) in enumerate(tqdm(testloader)):

        if is_cuda:
            inputs = inputs.cuda()

        en_img = model.encoder(img)
        en_ir = model.encoder(ir)

        f = model.fusion(en_img, en_ir, strategy_type=strategy_type)

        fused_img = model.decoder(f)

        if is_cuda:
            fused_img = fused_img.cpu()
        else:
            pass

        save_image(fused_img[0],args.test_save_dir + 'result_{}.png'.format(i))

    print('Finished testing')