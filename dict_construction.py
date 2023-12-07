# modify from https://github.com/TencentAILabHealthcare/Few-shot-WSI/blob/master/wsi_workdir/dict_construction.py
import argparse
import os
import numpy as np
import clustering
from pathlib import Path

def parse_option():
    parser = argparse.ArgumentParser('argument for testing')
    parser.add_argument('--feat_file', type=str, default='feature/NCT_pre@SimCLRTile_NCT_feat.npy')
    parser.add_argument('--output', type=str, default='./Dict/')
    parser.add_argument('--num_prototypes', type=int, default=16)
    parser.add_argument('--num_shift_vectors', type=int, default=2000)
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_option()
    
    features = np.load(opt.feat_file, 'r')

    feat_file = os.path.basename(opt.feat_file)
    pretrain_data = feat_file.split('_')[0]
    model = feat_file.split('_')[1]

    #os.makedirs(f'{opt.output}/{model}/', exist_ok=True)
    
    path = Path(f'{opt.output}/{model}/')
    path.mkdir(parents=True, exist_ok=True)

    kmeans = clustering.Kmeans(k=opt.num_prototypes, pca_dim=-1)
    kmeans.cluster(features, seed=66)
    assignments = kmeans.labels.astype(np.int64)
    
    prototypes = np.array([np.mean(features[assignments==i],axis=0)
                                     for i in range(opt.num_prototypes)])

    covariance = np.array([np.cov(features[assignments==i].T) 
                                    for i in range(opt.num_prototypes)])

    np.save(f'{opt.output}/{model}/{pretrain_data}_PROTO_BANK_{opt.num_prototypes}.npy', prototypes)
    np.save(f'{opt.output}/{model}/{pretrain_data}_COV_BANK_{opt.num_prototypes}.npy', covariance)

    SHIFT_BANK = []
    for cov in covariance:
        SHIFT_BANK.append(
                    # sample shift vector from zero-mean multivariate Gaussian distritbuion N(0, cov)
                    np.random.multivariate_normal(np.zeros(cov.shape[0]),
                    cov, 
                    size=opt.num_shift_vectors))

    SHIFT_BANK = np.array(SHIFT_BANK)
    # save the shift bank
    np.save(f'{opt.output}/{model}/{pretrain_data}_SHIFT_BANK_{opt.num_prototypes}.npy', SHIFT_BANK)
    print('legacy dict constructed', f'saving to {opt.output}/{model}/{pretrain_data}_SHIFT_BANK_{opt.num_prototypes}.npy')

main()
