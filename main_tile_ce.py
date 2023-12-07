from __future__ import print_function

import os
import sys
import argparse
import time
import math
import torch
import torch.backends.cudnn as cudnn
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.models import SupCEResNet
from WSI_tile_dataset import WSI_tile_dataset
from Tile_dataset import Tile_dataset
import augmentation

try:
    import apex  # noqa
    from apex import amp, optimizers  # noqa
except ImportError:
    pass

torch.manual_seed(0)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--base_learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='350,400,450',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--model', type=str, default='dinov2_vitb14')
    parser.add_argument('--dataset', type=str, default='TCGA',
                        choices=['TCGA', 'GTEx', 'NCT', 'LC'], help='dataset')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    opt.cosine = True
    # opt.syncBN = True
    opt.learning_rate = opt.batch_size * opt.base_learning_rate / 256
    # set the path according to the environment
    opt.model_path = './save/SupCETile/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCETile/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCETile_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'GTEx':
        opt.n_cls = 36
    elif opt.dataset == 'TCGA':
        opt.n_cls = 27
    elif opt.dataset == 'NCT':
        opt.n_cls = 9 
    elif opt.dataset == 'LC':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


def set_loader(opt):
    # construct data loader    
    train_transform = augmentation.get_aug(name='simsiam', image_size=224, 
                                     train=False, train_classifier=True)
    val_transform = augmentation.get_aug(name='simsiam', image_size=224, 
                                   train=False, train_classifier=False)

    if opt.dataset == 'GTEx':
        train_dataset = WSI_tile_dataset(
                tile_dir='/mnt/data3/renyu/GTEx/GTEx_Tiles_jpg/',
                label_file='/mnt/data3/renyu/GTEx/slide_split/train_slide_lt.csv',  # noqa
                transform=train_transform)
        val_dataset = WSI_tile_dataset(tile_dir='/mnt/data3/renyu/GTEx/GTEx_Tiles_jpg/',  # noqa
                label_file='/mnt/data3/renyu/GTEx/slide_split/val_slide_lt.csv',  # noqa
                transform=val_transform)  # noqa
    elif opt.dataset == 'TCGA':
        train_dataset = WSI_tile_dataset(tile_dir='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/TCGA_tiles/',  # noqa
                label_file='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/ProjectName_slide_split/train_slide_lt.csv',  # noqa
                transform=train_transform)
        val_dataset = WSI_tile_dataset(tile_dir='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/TCGA_tiles/',
                label_file='/mnt/data1/renyu/Few-shot-dataset/20X_224x224/TCGA_20X_224/ProjectName_slide_split/train_slide_lt.csv',
                transform=val_transform)
    elif opt.dataset == 'NCT':
        train_transform = augmentation.get_transform(dataset=opt.dataset, stage='simple_train', size=224)
        train_dataset = Tile_dataset('datasets/NCT/data/',
                'datasets/NCT/meta/train_labeled.txt',
                transform=train_transform,
                two_view=True
                )
        val_dataset = Tile_dataset('datasets/NCT/data/',
                'datasets/NCT/meta/test_labeled.txt',
                transform=val_transform,
                two_view=True)
    elif opt.dataset == 'LC':
        train_transform = augmentation.get_transform(dataset=opt.dataset, stage='simple_train', size=224)
        train_dataset = Tile_dataset('datasets/LC25000/data/',
                'datasets/LC25000/split/train_labeled.txt',
                transform=train_transform,
                two_view=True
                )
        val_dataset = Tile_dataset('datasets/LC25000/data/',
                'datasets/LC25000/split/test_labeled.txt',
                transform=val_transform,
                two_view=True)
    else:
        raise ValueError(opt.dataset)

    weight_lt = train_dataset.get_weight()
    train_sampler = torch.utils.data.WeightedRandomSampler(weight_lt, num_samples=(len(weight_lt)//opt.batch_size)*opt.batch_size)

    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=False,
            num_workers=opt.num_workers, pin_memory=True, drop_last=True)
  
    return train_loader, val_loader


def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()
    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    end = time.time()
    for idx, ((images, _), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc3 = accuracy(output, labels, topk=(1, 3))
        top1.update(acc1[0], bsz)
        top3.update(acc3[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print(f'Train: [{epoch}][{idx}/{len(train_loader)}] loss: {losses.val:.4f} ({losses.avg:.4f}) Acc@1: {top1.val:.3f} ({top1.avg:.3f}) Acc@3: {top3.val:.3f} ({top3.avg:.3f})') 
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, ((images, _), labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc3 = accuracy(output, labels, topk=(1, 3))
            top1.update(acc1[0], bsz)
            top3.update(acc3[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print(f'Validate: [{idx}/{len(val_loader)}] loss: {losses.val:.4f} ({losses.avg:.4f}) Acc@1: {top1.val:.3f} ({top1.avg:.3f}) Acc@3: {top3.val:.3f} ({top3.avg:.3f})')
    print(f' * Acc@1: {top1.avg:.4f} Acc@3: {top3.avg:.4f}')
    return losses.avg, top1.avg


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        loss, val_acc = validate(val_loader, model, criterion, opt)

        if val_acc > best_acc:
            best_acc = val_acc

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))


if __name__ == '__main__':
    main()
