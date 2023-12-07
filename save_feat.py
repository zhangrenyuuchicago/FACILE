import torch
import torch.nn as nn
import os
from Tile_dataset import Tile_dataset
from networks.models import SupConResNetWSI, SupCEResNetWSI
from networks.models import SupConResNet, SupCEResNet
import torchvision
from networks.simsiam import SimSiam
import numpy as np
import argparse
import augmentation
import tqdm
from sklearn.preprocessing import normalize

np.random.seed(0)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--model', type=str, default='dinov2_vitb14')
    #parser.add_argument('--dataset', type=str, default='LC')
    parser.add_argument('--dataset', type=str, default='NCT')
    #parser.add_argument('--dataset', type=str, default='PAIP')
    #parser.add_argument('--dataset', type=str, default='TCGA')
    parser.add_argument('--model_file', type=str, 
            default='DINOV2')
            #default='save/SimSiamTile/TCGA_1over20/TCGA_models/SimSiamTile_TCGA_resnet18_lr_0.0107421875_decay_0.0001_bsz_55_temp_0.07_trial_0_cosine/last.pth')
            #default='save/SimCLRTile/NCT_models/SimCLRTile_NCT_resnet18_lr_0.00625_decay_0.0001_bsz_32_temp_0.07_trial_0_cosine/last.pth')
            #default='save/SupCETile/TCGA_models/SupCETile_TCGA_resnet18_lr_0.05_decay_0.0001_bsz_64_trial_0_cosine/last.pth')
    parser.add_argument('--data_folder', type=str, default='../datasets/NCT/data/')
    parser.add_argument('--label_file', type=str, default='../datasets/NCT/meta/train_labeled.txt')
    #parser.add_argument('--data_folder', type=str, default='datasets/PAIP/data/')
    #parser.add_argument('--label_file', type=str, default='datasets/PAIP/meta/all_labels.txt')
    #parser.add_argument('--data_folder', type=str, default='../datasets/LC25000/data/')
    #parser.add_argument('--label_file', type=str, default='../datasets/LC25000/meta/img_list_labeled.txt')
    #parser.add_argument('--data_folder', type=str, default='datasets/PDAC/')
    #parser.add_argument('--label_file', type=str, default='datasets/PDAC/meta/all_labels.txt')
    #parser.add_argument('--data_folder', type=str, default='/mnt/data1/TCGA_20X_224/TCGA_tiles/')
    #parser.add_argument('--label_file', type=str, default='/mnt/data1/TCGA_20X_224/tile_split/sub_train_tile_lt_label_1over20.txt')
 
    parser.add_argument('--output', type=str, default='feature/')

    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()
    if os.path.exists(opt.output):
        print(f'output: {opt.output}; exists')
    else:
        os.mkdir(opt.output)
    
    if opt.model_file == 'ImageNet':
        print('use ImageNet pretrained weights')
        resnet = torchvision.models.resnet18(weights='DEFAULT')
        model = torch.nn.Sequential(*list(resnet.children())[:-1], nn.Flatten(start_dim=1))
        pretrain_model = 'SupCE'
        pretrain_dataset = 'ImageNet'
    elif opt.model_file == 'DINOV2':
        print('use DINOV2 pretrained weights')
        dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        model = dinov2_vitb14
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()

        pretrain_model = 'DINOV2'
        pretrain_dataset = 'DINOV2'

    else:
        print('use checkpoint weights')
        pretrain_model = os.path.basename(os.path.dirname(opt.model_file))
        pretrain_model = pretrain_model.split('_')[0]
        basename = os.path.basename(opt.model_file) 
        array = basename[:-4].split('_')
        opt.epoch = array[-1] 

        pretrain_dataset = os.path.basename(os.path.dirname(opt.model_file))
        pretrain_dataset = pretrain_dataset.split('_')[1]
   
        if pretrain_dataset == 'GTEx':
            opt.n_cls = 36
        elif pretrain_dataset == 'TCGA':
            opt.n_cls = 27
        elif pretrain_dataset == 'NCT':
            opt.n_cls = 9
        elif pretrain_dataset == 'LC':
            opt.n_cls = 5
        else:
            raise ValueError('pretrain dataset not supported: {}'.format(pretrain_dataset))

        if pretrain_model == 'SupConWSI' or pretrain_model == 'SimCLRWSI':
            #model = SupConResNetWSI(name=opt.model, head='linear')
            model = SupConResNetWSI(name=opt.model)
        elif pretrain_model == 'SupCEWSI':
            model = SupCEResNetWSI(name=opt.model, num_classes=opt.n_cls)
        elif pretrain_model == 'SupConTile' or pretrain_model == 'SimCLRTile':
            #model = SupConResNet(name=opt.model, head='linear')
            model = SupConResNet(name=opt.model)
        elif pretrain_model == 'SupCETile':
            model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
        elif pretrain_model == 'SimSiamTile':
            model = SimSiam(name=opt.model) 
        else:
            raise ValueError(f'pretrain model not found: {pretrain_model}')

        state = torch.load(opt.model_file)['model']
        update_state = {key.replace('.module', ''):state[key] for key in state}
        update_state = {key.replace('module.', ''):state[key] for key in state}

        model.load_state_dict(update_state)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        if pretrain_model == 'SimSiamTile':
            model = model.encoder
        else:
            print(f'possible bugs')
            model = model.encoder

    transform = augmentation.get_aug(name='simsiam', image_size=224, train=False, train_classifier=False)
    tile_dataset = Tile_dataset(opt.data_folder, opt.label_file, transform=transform)

    params = {'batch_size': 64,
                'shuffle': False,
                'num_workers': 16}
    tile_dataloader = torch.utils.data.DataLoader(tile_dataset, **params)

    print(f'pretrain model: {pretrain_model}')
    print(f'pretrain dataset: {pretrain_dataset}')
    print(f'generate feature for dataset: {opt.dataset}')

    if tile_dataset.with_label: 
        feat_lt = []
        label_lt = []
        for idx, (image, label) in enumerate(tqdm.tqdm(tile_dataloader)):
            if torch.cuda.is_available():
                image = image.cuda()
                label = label.cuda()

            feat = model(image)
            feat_lt += list(feat.data.cpu().numpy())
            label_lt += list(label.data.cpu().numpy())

        feat_lt = np.array(feat_lt)
        feat_lt = normalize(feat_lt, axis=1, norm='l2')

        label_lt = np.array(label_lt)

        image_feat_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_feat_epoch{opt.epoch}.npy')
        label_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_label_epoch{opt.epoch}.npy')
        #image_feat_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_feat.npy')
        #label_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_label.npy')
 
        np.save(image_feat_path, feat_lt)
        np.save(label_path, label_lt)
    else:
        feat_lt = []
        for idx, image in enumerate(tqdm.tqdm(tile_dataloader)):
            if torch.cuda.is_available():
                image = image.cuda()

            feat = model(image)
            feat_lt += list(feat.data.cpu().numpy())

        feat_lt = np.array(feat_lt)
        feat_lt = normalize(feat_lt, axis=1, norm='l2')
        
        image_feat_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_feat_epoch{opt.epoch}.npy')
        #image_feat_path = os.path.join(opt.output, f'{opt.dataset}_pre@{pretrain_model}_{pretrain_dataset}_feat.npy')

        np.save(image_feat_path, feat_lt)


main()
