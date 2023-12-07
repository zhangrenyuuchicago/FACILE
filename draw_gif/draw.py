import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def main():
    feat_file_lt = [f'../feature/NCT_pre@SupCEWSI_TCGA_feat_epoch{i}.npy' for i in range(1,11)]
    label_file_lt = [f'../feature/NCT_pre@SupCEWSI_TCGA_label_epoch{i}.npy' for i in range(1,11)]

    print(feat_file_lt)
    print(label_file_lt)

    feat0 = np.load(feat_file_lt[0])
    label0 = np.load(label_file_lt[0])

    feat_lt = []
    for feat_file, label_file in zip(feat_file_lt, label_file_lt):
        feat = np.load(feat_file)
        label = np.load(label_file)

        assert all(label == label0)
        feat_lt.append(feat[0:1000])

    feat_lt = np.concatenate(feat_lt, axis=0)
    label_lt = label0[0:1000]
    
    label2str = {0: 'BACK', 1: 'ADI', 2: 'DEB', 3: 'LYM', 4: 'MUC', 5: 'MUS', 6: 'NORM', 7: 'STR', 8: 'TUM'}

    # use t-SNE to reduce dimension
    tsne = TSNE(n_components=2, random_state=0)
    feat_lt = tsne.fit_transform(feat_lt)
    
    for i in range(len(feat_file_lt)):
        start = i * 1000
        end = (i + 1) * 1000
        feat = feat_lt[start:end]
        unique_labels = set(label_lt)
        for label in unique_labels:
            x = [point[0] for point, l in zip(feat, label_lt) if l == label]
            y = [point[1] for point, l in zip(feat, label_lt) if l == label]

            plt.scatter(x, y, label=f"{label2str[label]}", s=5)

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # legend out of plot
        plt.subplots_adjust(right=0.8)
        plt.title(f'epoch {i+1}')
        plt.savefig('tsne_epoch%d.png' % (i+1))
        plt.close()

if __name__ == '__main__':
    main()
        