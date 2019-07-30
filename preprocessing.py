import os
import numpy as np
from scipy.io import loadmat


radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1',
                 'gfsk1', 'pam4', '16qam', '64qam']

diagram_classes = ['16qam', '64qam']


def load_data_from_mat(data_path, snr_index, test_split=0.3):
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for index, src in enumerate(radio_classes):
        if isinstance(src, list):
            for s in src:
                dat = np.zeros((2, 6000, 1))
                for i, typ in enumerate(['im', 're']):
                    path = os.path.join(data_path, 'src_{}_{}.mat'.format(s, typ))
                    dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(s, typ)][snr_index]
                split = int(6000 * (1 - test_split))
                for i in range(split - 127):
                    train_data.append(dat[:, i:i + 128, :])
                    train_label.append(index)
                for i in range((6000-split-128)/128):
                    test_data.append(dat[:, split + i * 128:split + (i + 1) * 128, :])
                    test_label.append(index)
                # train_ind = int(6000 / 128 * (1 - test_split))
                # for i in range(6000 / 128):
                #     if i > train_ind - 1:
                #         test_data.append(dat[:, i * 128:(i + 1) * 128, :])
                #         test_label.append(index)
                #     else:
                #         train_data.append(dat[:, i * 128:(i + 1) * 128, :])
                #         train_label.append(index)

        else:
            dat = np.zeros((2, 6000, 1))
            for i, typ in enumerate(['im', 're']):
                path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
                dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][snr_index]
            split = int(6000 * (1 - test_split))
            for i in range(split - 127):
                train_data.append(dat[:, i:i + 128, :])
                train_label.append(index)
            for i in range((6000 - split - 128) / 128):
                test_data.append(dat[:, split + i * 128:split + (i + 1) * 128, :])
                test_label.append(index)
            # train_ind = int(6000/128 * (1 - test_split))
            # for i in range(6000/128):
            #     if i > train_ind - 1:
            #         test_data.append(dat[:, i * 128:(i + 1) * 128, :])
            #         test_label.append(index)
            #     else:
            #         train_data.append(dat[:, i * 128:(i + 1) * 128, :])
            #         train_label.append(index)
        print np.array(train_data).shape
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

if __name__ == '__main__':
    load_data_from_mat('datasets/signal_dataset', 0)
