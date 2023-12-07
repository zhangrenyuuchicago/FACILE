from __future__ import print_function

import os
import sys
import argparse
import time
import torch
import torch.backends.cudnn as cudnn
from util import AverageMeter
from util import warmup_learning_rate
from util import set_optimizer, save_model
from networks.simsiam import SimSiam
from WSI_tile_dataset import WSI_tile_dataset
import augmentation
from torch.optim.lr_scheduler import CosineAnnealingLR
from Tile_dataset import Tile_dataset

try:
    import apex # noqa
    from apex import amp, optimizers # noqa
except ImportError:
    pass

torch.manual_seed(0)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of training epochs')
    # optimization
    parser.add_argument('--base_learning_rate', type=float, default=0.05,
                        help='learning rate')
    
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # model dataset
    parser.add_argument('--model', type=str, default='dinov2_vitb14')
    parser.add_argument('--dataset', type=str, default='TCGA',
                        choices=['TCGA', 'GTEx', 'PDAC'], help='dataset')
    parser.add_argument('--method', type=str, default='SimSiamTile',
                        choices=['SimSiamTile'], help='choose method')
    # temperature
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()
    opt.cosine = True
    opt.learning_rate = opt.base_learning_rate * opt.batch_size / 256 

    assert opt.method == 'SimSiamTile'
    # set the path according to the environment
    opt.model_path = './save/SimSiamTile/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SimSiamTile/{}_tensorboard'.format(opt.dataset)

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_loader(opt):
    # construct data loader
    transform = augmentation.get_aug(name='simsiam', image_size=224, train=True, train_classifier=None)

    if opt.dataset == 'TCGA':
        train_dataset = WSI_tile_dataset(tile_dir='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/TCGA_tiles/',
                label_file='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/ProjectName_slide_split/train_slide_lt.csv',
                transform=transform)
    elif opt.dataset == 'GTEx':
        train_dataset = WSI_tile_dataset(tile_dir='/mnt/data3/renyu/GTEx/GTEx_Tiles_jpg',
                label_file='/mnt/data3/renyu/GTEx/slide_split/train_slide_lt.csv',
                transform=transform)
    elif opt.dataset == 'NCT':
        train_dataset = Tile_dataset('../datasets/NCT/data/', '../datasets/NCT/meta/train_labeled.txt', transform=transform, two_view=True)
        train_sampler = torch.utils.data.RandomSampler(train_dataset, num_samples=6000)
        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        return train_loader
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SimSiam(name=opt.model)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)
        # currently only support Distributated
        #model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        cudnn.benchmark = True

    return model 


def train(train_loader, model, optimizer, lr_scheduler, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = labels.shape[0]
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        image1, image2 = images[0], images[1]
        if torch.cuda.is_available():
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
 
        loss = model(image1, image2)
        losses.update(loss.item(), bsz)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def main():
    opt = parse_option()
    # build data loader
    train_loader = set_loader(opt)
    iters = len(train_loader)
    # build model and criterion
    model = set_model(opt)
    optimizer = set_optimizer(opt, model)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=opt.epochs*iters)
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, optimizer, lr_scheduler, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == '__main__':
    main()

