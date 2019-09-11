"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import torchvision
from tensorboardX import SummaryWriter
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common.args import Args
from common.subsample import MaskFunc
from data import transforms
from data.mri_data import SliceData
from models.unet.unet_model import UnetModel
from metrics.custom_losses import L1CSSIM
from models.discriminator.discriminators import define_Dis
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(self, mask_func, resolution, which_challenge, use_seed=True):
        """
        Args:
            mask_func (common.subsample.MaskFunc): A function that can create a mask of
                appropriate shape.
            resolution (int): Resolution of the image.
            which_challenge (str): Either "singlecoil" or "multicoil" denoting the dataset.
            use_seed (bool): If true, this class computes a pseudo random number generator seed
                from the filename. This ensures that the same mask is used for all the slices of
                a given volume every time.
        """
        if which_challenge not in ('singlecoil', 'multicoil'):
            raise ValueError(f'Challenge should either be "singlecoil" or "multicoil"')
        self.mask_func = mask_func
        self.resolution = resolution
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(self, kspace, target, attrs, fname, slice):
        """
        Args:
            kspace (numpy.array): Input k-space of shape (num_coils, rows, cols, 2) for multi-coil
                data or (rows, cols, 2) for single coil data.
            target (numpy.array): Target image
            attrs (dict): Acquisition related information stored in the HDF5 object.
            fname (str): File name
            slice (int): Serial number of the slice.
        Returns:
            (tuple): tuple containing:
                image (torch.Tensor): Zero-filled input image.
                target (torch.Tensor): Target image converted to a torch Tensor.
                mean (float): Mean value used for normalization.
                std (float): Standard deviation value used for normalization.
                norm (float): L2 norm of the entire volume.
        """
        kspace = transforms.to_tensor(kspace)
        # Apply mask
        seed = None if not self.use_seed else tuple(map(ord, fname))
        masked_kspace, mask = transforms.apply_mask(kspace, self.mask_func, seed)
        # Inverse Fourier Transform to get zero filled solution
        image = transforms.ifft2(masked_kspace)
        # Crop input image
        image = transforms.complex_center_crop(image, (self.resolution, self.resolution))
        # Absolute value
        image = transforms.complex_abs(image)
        # Apply Root-Sum-of-Squares if multicoil data
        if self.which_challenge == 'multicoil':
            image = transforms.root_sum_of_squares(image)
        # Normalize input
        image, mean, std = transforms.normalize_instance(image, eps=1e-11)
        image = image.clamp(-6, 6)

        target = transforms.to_tensor(target)
        # Normalize target
        target = transforms.normalize(target, mean, std, eps=1e-11)
        target = target.clamp(-6, 6)
        return image, target, mean, std, attrs['norm'].astype(np.float32)


def create_datasets(args):
    train_mask = MaskFunc(args.center_fractions, args.accelerations)
    dev_mask = MaskFunc(args.center_fractions, args.accelerations)

    train_data = SliceData(
        root=args.data_path / f'{args.challenge}_train',
        transform=DataTransform(train_mask, args.resolution, args.challenge),
        sample_rate=args.sample_rate,
        challenge=args.challenge
    )
    dev_data = SliceData(
        root=args.data_path / f'{args.challenge}_val',
        transform=DataTransform(dev_mask, args.resolution, args.challenge, use_seed=True),
        sample_rate=args.sample_rate,
        challenge=args.challenge,
    )
    return dev_data, train_data


def create_data_loaders(args):
    dev_data, train_data = create_datasets(args)
    display_data = [dev_data[i] for i in range(0, len(dev_data), len(dev_data) // 16)]

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dataset=dev_data,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
    )
    display_loader = DataLoader(
        dataset=display_data,
        batch_size=16,
        num_workers=8,
        pin_memory=True,
    )
    return train_loader, dev_loader, display_loader

def set_grad(nets, requires_grad=False):
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grad

def train_epoch(args, epoch, model, model_dis, data_loader, optimizer, optimizer_dis, gen_loss_func, gan_loss_func, writer):
    model.train()
    model_dis.train()
    avg_loss = 0.
    target_real = torch.tensor(1, dtype=torch.float32, device=args.device)
    target_fake = torch.tensor(0, dtype=torch.float32, device=args.device)
    start_epoch = start_iter = time.perf_counter()
    global_step = epoch * len(data_loader)
    for iter, data in enumerate(data_loader):
        input, target, mean, std, norm = data
        if input.size(0)< args.batch_size: 
            continue
        input = input.unsqueeze(1).to(args.device)
        target = target.to(args.device)    
        
        # Forward input to generator
        ############################        
        output = model(input) #.squeeze(1)
        # Train D network
        ############################
        set_grad([model_dis], True)  # enable backprop for D
        optimizer_dis.zero_grad()     # set D's gradients to zero
        # Fake
        fake_AB = output #torch.cat((input, output), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = model_dis(fake_AB.detach())
        target_fake = target_fake.expand_as(pred_fake) 
        loss_D_fake = gan_loss_func(pred_fake, target_fake)
        # Real
        
        real_AB = target.unsqueeze(1) #torch.cat((input, target.unsqueeze(1)), 1)
        pred_real = model_dis(real_AB)
        target_real = target_real.expand_as(pred_real)
        loss_D_real = gan_loss_func(pred_real, target_real)
        # combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_D.backward()
        optimizer_dis.step()          # update D's weights

        
        # Train D network
        ############################
        set_grad([model_dis], True)  # enable backprop for D  # D requires no gradients when optimizing G
        optimizer.zero_grad()        # set G's gradients to zero
        # calculate graidents for G
        # First, G(A) should fake the discriminator
        fake_AB = output #torch.cat((input, output), 1)
        pred_fake = model_dis(fake_AB)
        target_real = target_real.expand_as(pred_fake)
        loss_G_GAN = gan_loss_func(pred_fake, target_real)
        # Second, G(A) = B
        output = output.squeeze(1)
        loss_G_recon = gen_loss_func(output, target)        
        # combine loss and calculate gradients
        loss = loss_G_GAN + loss_G_recon    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



        avg_loss = 0.99 * avg_loss + 0.01 * loss.item() if iter > 0 else loss.item()
        writer.add_scalar('TrainLoss', loss.item(), global_step + iter)

        if iter % args.report_interval == 0:
            logging.info(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss G = {loss_G_recon.item():.4g} Loss D = {loss_G_GAN.item():.4g}  Loss = {loss.item():.4g}  Avg Loss = {avg_loss:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
        start_iter = time.perf_counter()
    return avg_loss, time.perf_counter() - start_epoch


def evaluate(args, epoch, model, data_loader, writer):
    model.eval()
    losses = []
    start = time.perf_counter()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.to(args.device)
            output = model(input).squeeze(1)

            mean = mean.unsqueeze(1).unsqueeze(2).to(args.device)
            std = std.unsqueeze(1).unsqueeze(2).to(args.device)
            target = target * std + mean
            output = output * std + mean

            norm = norm.unsqueeze(1).unsqueeze(2).to(args.device)
            loss = F.mse_loss(output / norm, target / norm, size_average=False)
            losses.append(loss.item())
        writer.add_scalar('Dev_Loss', np.mean(losses), epoch)
    return np.mean(losses), time.perf_counter() - start


def visualize(args, epoch, model, data_loader, writer):
    def save_image(image, tag):
        image -= image.min()
        image /= image.max()
        grid = torchvision.utils.make_grid(image, nrow=4, pad_value=1)
        writer.add_image(tag, grid, epoch)

    model.eval()
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            input, target, mean, std, norm = data
            input = input.unsqueeze(1).to(args.device)
            target = target.unsqueeze(1).to(args.device)
            output = model(input)
            save_image(target, 'Target')
            save_image(output, 'Reconstruction')
            save_image(torch.abs(target - output), 'Error')
            break


def save_model(args, exp_dir, epoch, model, model_dis, optimizer,optimizer_dis, best_dev_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model_dis': model_dis.state_dict(),
            'optimizer_dis': optimizer_dis.state_dict(),
            'best_dev_loss': best_dev_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / 'model.pt', exp_dir / 'best_model.pt')


def build_model(args):
    model = UnetModel(
        in_chans=1,
        out_chans=1,
        chans=args.num_chans,
        num_pool_layers=args.num_pools,
        drop_prob=args.drop_prob
    ).to(args.device)
    model_dis = define_Dis(input_nc=1, ndf=64, netD= 'n_layers', n_layers_D=5, norm='instance').to(args.device)
        
    return model, model_dis


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    args = checkpoint['args']
    model, model_dis = build_model(args)
    if args.data_parallel:
        model = torch.nn.DataParallel(model)
        model_dis= torch.nn.DataParallel(model_dis)
    model.load_state_dict(checkpoint['model'])
    #model_dis.load_state_dict(checkpoint['model_dis'])

    optimizer,optimizer_dis  = build_optim(args, model.parameters(), model_dis.parameters())
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #optimizer_dis.load_state_dict(checkpoint['optimizer_dis'])
    return checkpoint, model,model_dis, optimizer, optimizer_dis


def build_optim(args, params, params_dis):
    optimizer = torch.optim.Adam(params, args.lr, weight_decay=args.weight_decay)
    optimizer_dis = torch.optim.Adam(params_dis,lr=2e-4, betas=(0.5, 0.999))
    return optimizer, optimizer_dis


def main(args):
    args.exp_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.exp_dir / 'summary')

    if args.resume:
        print ('--------------Resume ---------------------')
        checkpoint, model, model_dis, optimizer, optimizer_dis = load_model(args.checkpoint)
        args = checkpoint['args']
        best_dev_loss = checkpoint['best_dev_loss']
        start_epoch = checkpoint['epoch']
        del checkpoint
    else:
        model, model_dis = build_model(args)
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
            model_dis = torch.nn.DataParallel(model_dis)
        optimizer, optimizer_dis = build_optim(args, model.parameters(), model_dis.parameters())
        best_dev_loss = 1e9
        start_epoch = 0
    logging.info(args)
    logging.info(model)
    logging.info(optimizer_dis)

    train_loader, dev_loader, display_loader = create_data_loaders(args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gamma)
    scheduler_dis = torch.optim.lr_scheduler.StepLR(optimizer_dis, args.lr_step_size, args.lr_gamma)
    recon_loss_func = L1CSSIM(l1_weight=1.0, default_range=12, filter_size=7, reduction='mean')
    gan_loss_func = torch.nn.BCEWithLogitsLoss()
    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)
        scheduler_dis.step(epoch)
        train_loss, train_time = train_epoch(args, epoch, model, model_dis, train_loader, optimizer, optimizer_dis, recon_loss_func, gan_loss_func, writer)
        dev_loss, dev_time = evaluate(args, epoch, model, dev_loader, writer)
        visualize(args, epoch, model, display_loader, writer)

        is_new_best = dev_loss < best_dev_loss
        best_dev_loss = min(best_dev_loss, dev_loss)
        save_model(args, args.exp_dir, epoch, model, model_dis, optimizer, optimizer_dis, best_dev_loss, is_new_best)
        logging.info(
            f'Epoch = [{epoch:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'DevLoss = {dev_loss:.4g} TrainTime = {train_time:.4f}s DevTime = {dev_time:.4f}s',
        )
    writer.close()


def create_arg_parser():
    parser = Args()
    parser.add_argument('--num-pools', type=int, default=4, help='Number of U-Net pooling layers')
    parser.add_argument('--drop-prob', type=float, default=0.0, help='Dropout probability')
    parser.add_argument('--num-chans', type=int, default=32, help='Number of U-Net channels')

    parser.add_argument('--batch-size', default=32, type=int, help='Mini batch size')
    parser.add_argument('--num-epochs', type=int, default=45, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr-step-size', type=int, default=15,
                        help='Period of learning rate decay')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='Multiplicative factor of learning rate decay')
    parser.add_argument('--weight-decay', type=float, default=0.,
                        help='Strength of weight decay regularization')

    parser.add_argument('--report-interval', type=int, default=100, help='Period of loss reporting')
    parser.add_argument('--data-parallel', action='store_true',
                        help='If set, use multiple GPUs using data parallelism')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Which device to train on. Set to "cuda" to use the GPU')
    parser.add_argument('--exp-dir', type=pathlib.Path, default='checkpoints',
                        help='Path where model and results should be saved')
    parser.add_argument('--resume', action='store_true',
                        help='If set, resume the training from a previous model checkpoint. '
                             '"--checkpoint" should be set with this')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to an existing checkpoint. Used along with "--resume"')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    main(args)
