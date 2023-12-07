import random
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import NearestCentroid
from sklearn.metrics import f1_score, accuracy_score
from tqdm.contrib.concurrent import process_map
from scipy.spatial.distance import cdist
import scipy
from scipy.stats import t

random.seed(0)
np.random.seed(0)

def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    #parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@SupCE_ImageNet_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/NCT_pre@SupCE_ImageNet_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@DINOV2_DINOV2_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/NCT_pre@DINOV2_DINOV2_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@SimSiamTile_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/NCT_pre@SimSiamTile_TCGA_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@SimCLRTile_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/NCT_pre@SimCLRTile_TCGA_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@SupConWSI_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/NCT_pre@SupConWSI_TCGA_label.npy')
    parser.add_argument('--feat_path', type=str, default='feature/NCT_pre@SupCEWSI_TCGA_feat.npy')
    parser.add_argument('--label_path', type=str, default='feature/NCT_pre@SupCEWSI_TCGA_label.npy')
 
    #parser.add_argument('--feat_path', type=str, default='feature/LC_pre@SimSiamTile_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/LC_pre@SimSiamTile_TCGA_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/LC_pre@SimCLRTile_NCT_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/LC_pre@SimCLRTile_NCT_label.npy') 
    #parser.add_argument('--feat_path', type=str, default='feature/ICLR_feat/clp/LC.npy')
    #parser.add_argument('--label_path', type=str, default='feature/ICLR_feat/clp/labels.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/PAIP_pre@SupCETile_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/PAIP_pre@SupCETile_TCGA_label.npy')
    #parser.add_argument('--feat_path', type=str, default='feature/PAIP/PAIP_pre@SupCEWSI_TCGA_feat.npy')
    #parser.add_argument('--label_path', type=str, default='feature/PAIP/PAIP_pre@SupCEWSI_TCGA_label.npy')
    #parser.add_argument('--proto_bank_path', type=str, default='Dict/pre@SimCLRTile/NCT_PROTO_BANK_16.npy')
    #parser.add_argument('--shift_bank_path', type=str, default='Dict/pre@SimCLRTile/NCT_SHIFT_BANK_16.npy')
    #parser.add_argument('--proto_bank_path', type=str, default='Dict/pre@SimSiamTile/TCGA_PROTO_BANK_16.npy')
    #parser.add_argument('--shift_bank_path', type=str, default='Dict/pre@SimSiamTile/TCGA_SHIFT_BANK_16.npy')


    parser.add_argument('--clf_name', type=str, default='LogisticRegression')
    parser.add_argument('--task_num', type=int, default=1000)
    parser.add_argument('--num_way', type=int, default=9)
    parser.add_argument('--num_shot', type=int, default=5)
    parser.add_argument('--num_query', type=int, default=15)
    parser.add_argument('--num_aug_shot', type=int, default=100)
    parser.add_argument('--LA', action='store_true', help='Using Latent augmentation')

    opt = parser.parse_args()

    return opt

def gen_task(feats, labels, num_way, num_shot, num_query):
    class_num = np.unique(labels).shape[0]
    label_set = list(np.unique(labels))
    assert num_way <= class_num

    support_xs, support_ys = [], []
    query_xs, query_ys = [], []
    support_idxs, query_idxs = [], []

    #label_sampled_lt = random.sample([i for i in range(class_num)], num_way)
    #label_idxs = [np.where(labels==label)[0] for label in range(class_num)]
    label_sampled_lt = random.sample(label_set, num_way)
    label_idxs = [np.where(labels==label)[0] for label in label_sampled_lt]

    #for label in label_sampled_lt:
    for label_index in range(len(label_sampled_lt)):
        _all_idxs = np.random.choice(label_idxs[label_index], num_shot + num_query, replace=False)
        _support_idxs = _all_idxs[:num_shot]
        _query_idxs = _all_idxs[num_shot:]

        support_idxs.append(_support_idxs)
        query_idxs.append(_query_idxs)

        support_xs.append(feats[_support_idxs])
        #support_ys.append(labels[_support_idxs])
        support_ys.append(np.full((num_shot,), label_index, dtype=int))

        query_xs.append(feats[_query_idxs])
        #query_ys.append(labels[_query_idxs])
        query_ys.append(np.full((num_query,), label_index, dtype=int))

    support_xs = np.concatenate(support_xs)
    support_ys = np.concatenate(support_ys)
    query_xs = np.concatenate(query_xs)
    query_ys = np.concatenate(query_ys)

    support_idxs = np.concatenate(support_idxs)
    query_idxs = np.concatenate(query_idxs)

    return support_xs, support_ys, query_xs, query_ys

def aug_base_samples(features, labels, PROTO_BANK, SHIFT_BANK, NUM_SHIFTs, num_aug_shots=50):
    samples, gt_labels = [], []
    # for each class
    for label in np.unique(labels):
        selected_samples = features[labels==label]
        # find the most closest prototypes
        proto_id = np.argmin(cdist(selected_samples, PROTO_BANK), axis=1)
        generated_samples = []
        for ix, sample in zip(proto_id, selected_samples):
            generated_samples.append(
                np.concatenate([
                    [sample],
                    # generate new latent augmented samples by z' = z + delta
                    # delta is sampled from pre-generated shift bank, indexed by prototype id.
                    sample[np.newaxis, :] + SHIFT_BANK[ix][np.random.choice(NUM_SHIFTs,num_aug_shots)]
                ])
            )
        samples.append(np.concatenate(generated_samples, axis=0))
        gt_labels.extend([label]*len(samples[-1]))
    return np.concatenate(samples, axis=0), gt_labels

def meta_testing(args):
    feats = args['feats']
    labels = args['labels']
    
    def shuffle_in_unison(a, b):
        assert len(a) == len(b)
        shuffled_a = np.empty(a.shape, dtype=a.dtype)
        shuffled_b = np.empty(b.shape, dtype=b.dtype)
        permutation = np.random.permutation(len(a))
        for old_index, new_index in enumerate(permutation):
            shuffled_a[new_index] = a[old_index]
            shuffled_b[new_index] = b[old_index]
        return shuffled_a, shuffled_b  


    support_xs, support_ys, query_xs, query_ys = gen_task(feats, labels,
                args['num_way'], args['num_shot'], args['num_query'])
    
    
    #support_xs, support_ys = shuffle_in_unison(support_xs, support_ys)
    #print('************************')
    #print(f'support xs: {support_xs.shape}')
    #print(f'support ys: {support_ys}')

    if args['LA']:
        PROTO_BANK = args['PROTO_BANK']
        SHIFT_BANK = args['SHIFT_BANK']
        NUM_SHIFTs = len(SHIFT_BANK[0])

        support_xs, support_ys = aug_base_samples(support_xs,
                support_ys, PROTO_BANK, SHIFT_BANK, NUM_SHIFTs,
                num_aug_shots=args['num_aug_shot'])
        
        #print(f'support xs: {support_xs.shape}')
        #print(f'support ys: {support_ys}')

    if args['clf_name'] == 'RidgeClassifier':
        clf = RidgeClassifier()
    elif args['clf_name'] == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000)
    elif args['clf_name'] == 'NearestCentroid':
        clf = NearestCentroid()
    else:
        raise Exception(f'{clf_name} not implemented')

    clf.fit(support_xs, support_ys)
    y_pred = clf.predict(query_xs)
    #acc = accuracy_score(query_ys, y_pred, normalize=True)
    #f1 = f1_score(query_ys, y_pred, average=None)
    return query_ys, y_pred

def evaluate(y_trues, y_preds):
    f1s = []
    for y_true, y_pred in zip(y_trues, y_preds):
        f1s.append(f1_score(y_true, y_pred, average=None))
    return np.array(f1s)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h

def main_multi_thread():
    opt = parse_option()
    feats = np.load(opt.feat_path)
    labels = np.load(opt.label_path)

    if opt.LA:
        PROTO_BANK = np.load(opt.proto_bank_path)
        SHIFT_BANK = np.load(opt.shift_bank_path)
        NUM_SHIFTs = len(SHIFT_BANK[0])

        args = {'feats': feats,
            'labels': labels,
            'PROTO_BANK': PROTO_BANK,
            'SHIFT_BANK': SHIFT_BANK,
            'NUM_SHIFTs': NUM_SHIFTs,
            'clf_name': opt.clf_name,
            'task_num': opt.task_num,
            'num_way': opt.num_way,
            'num_shot': opt.num_shot,
            'num_query': opt.num_query,
            'num_aug_shot': opt.num_aug_shot,
            'LA': opt.LA
            }
    else:
        args = {'feats': feats,
            'labels': labels,
            'clf_name': opt.clf_name,
            'task_num': opt.task_num,
            'num_way': opt.num_way,
            'num_shot': opt.num_shot,
            'num_query': opt.num_query,
            'num_aug_shot': opt.num_aug_shot,
            'LA': opt.LA
            }


    acc_lt, f1_lt = [], []
    if args['LA']:
        print(f'test with LA')
        results = process_map(meta_testing, [args for i in range(opt.task_num)], max_workers=10, chunksize=5)
    else:
        results = process_map(meta_testing, [args for i in range(opt.task_num)], max_workers=40, chunksize=20)

    preds = np.array([x[1] for x in results])
    trues = np.array([x[0] for x in results])
    f1s = evaluate(trues, preds)

    means, cis = [], []
    for f1 in np.transpose(f1s,(1,0)):
        m, h = mean_confidence_interval(f1)
        means.append(m)
        cis.append(h)

    print(f'clf_name: {opt.clf_name}, num_way: {opt.num_way}, num_shot: {opt.num_shot}, num_query: {opt.num_query}, LA: {opt.LA}')
    #for m, h in zip(means, cis):
    #    print(f'{m*100:.2f} {h*100:.2f}')
    print(f'{np.mean(means)*100:.2f} +- {np.mean(cis)*100:.2f}')

def main():
    opt = parse_option()
    feats = np.load(opt.feat_path)
    labels = np.load(opt.label_path)

    if opt.LA:
        PROTO_BANK = np.load(opt.proto_bank_path)
        SHIFT_BANK = np.load(opt.shift_bank_path)
        NUM_SHIFTs = len(SHIFT_BANK[0])

        args = {'feats': feats,
            'labels': labels,
            'PROTO_BANK': PROTO_BANK,
            'SHIFT_BANK': SHIFT_BANK,
            'NUM_SHIFTs': NUM_SHIFTs,
            'clf_name': opt.clf_name,
            'task_num': opt.task_num,
            'num_way': opt.num_way,
            'num_shot': opt.num_shot,
            'num_query': opt.num_query,
            'num_aug_shot': opt.num_aug_shot,
            'LA': opt.LA
            }
    else:
        args = {'feats': feats,
            'labels': labels,
            'clf_name': opt.clf_name,
            'task_num': opt.task_num,
            'num_way': opt.num_way,
            'num_shot': opt.num_shot,
            'num_query': opt.num_query,
            'num_aug_shot': opt.num_aug_shot,
            'LA': opt.LA
            }


    acc_lt, f1_lt = [], []
    results = []
    '''
    if args['LA']:
        print(f'test with LA')
        results = process_map(meta_testing, [args for i in range(opt.task_num)], max_workers=10, chunksize=5)
    else:
        results = process_map(meta_testing, [args for i in range(opt.task_num)], max_workers=40, chunksize=20)
    '''
    for i in tqdm(range(opt.task_num)):
        ret = meta_testing(args)
        results.append(ret)

    preds = np.array([x[1] for x in results])
    trues = np.array([x[0] for x in results])
    f1s = evaluate(trues, preds)

    means, cis = [], []
    for f1 in np.transpose(f1s,(1,0)):
        m, h = mean_confidence_interval(f1)
        print(f'\t{m*100:.2f} +- {h*100:.2f}')
        means.append(m)
        cis.append(h)

    print(f'clf_name: {opt.clf_name}, num_way: {opt.num_way}, num_shot: {opt.num_shot}, num_query: {opt.num_query}, LA: {opt.LA}')
    print(f'{np.mean(means)*100:.2f} +- {np.mean(cis)*100:.2f}')

main()


