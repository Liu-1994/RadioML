import os
import numpy as np
from scipy.io import loadmat
from PIL import Image

radio_classes = ['qpsk', '8psk', 'bpsk', 'cpfsk1',
                 'gfsk1', 'pam4', '16qam', '64qam']

diagram_classes = ['16qam', '64qam']


def load_train_val_data_from_mat(data_path, snr_index=None, test_split=0.3):
    if snr_index or snr_index < 1:
        snrr_index = [snr_index]
    else:
        snrr_index = range(14)
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    points = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape[1]
    for index, src in enumerate(radio_classes):
        if isinstance(src, list):
            for s in src:
                for ss in snrr_index:
                    dat = np.zeros((2, points, 1))
                    for i, typ in enumerate(['im', 're']):
                        path = os.path.join(data_path, 'src_{}_{}.mat'.format(s, typ))
                        dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(s, typ)][ss]

                    # split = int(points * (1 - test_split))
                    # for i in range(split - 127):
                    #     train_data.append(dat[:, i:i + 128, :])
                    #     train_label.append(index)
                    # for i in range((points-split)/128):
                    #     test_data.append(dat[:, split + i * 128:split + (i + 1) * 128, :])
                    #     test_label.append(index)

                    train_ind = int(points / 128 * (1 - test_split))
                    for i in range(points / 128):
                        if i > train_ind - 1:
                            test_data.append(dat[:, i * 128:(i + 1) * 128, :])
                            test_label.append(index)
                        else:
                            train_data.append(dat[:, i * 128:(i + 1) * 128, :])
                            train_label.append(index)

        else:
            for ss in snrr_index:
                dat = np.zeros((2, points, 1))
                for i, typ in enumerate(['im', 're']):
                    path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
                    dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][ss]

                # split = int(points * (1 - test_split))
                # for i in range(split - 127):
                #     train_data.append(dat[:, i:i + 128, :])
                #     train_label.append(index)
                # for i in range((points - split) / 128):
                #     test_data.append(dat[:, split + i * 128:split + (i + 1) * 128, :])
                #     test_label.append(index)

                train_ind = int(points/128 * (1 - test_split))
                for i in range(points/128):
                    if i > train_ind - 1:
                        test_data.append(dat[:, i * 128:(i + 1) * 128, :])
                        test_label.append(index)
                    else:
                        train_data.append(dat[:, i * 128:(i + 1) * 128, :])
                        train_label.append(index)

        print np.array(train_data).shape
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def load_data_from_mat(data_path, snr_index=None):
    if snr_index or snr_index < 1:
        snrr_index = [snr_index]
    else:
        snrr_index = range(14)
    data = []
    label = []
    points = loadmat(os.path.join(data_path, 'src_bpsk_im.mat'))['src_bpsk_im'].shape[1]
    print points
    for index, src in enumerate(radio_classes):
        if isinstance(src, list):
            for s in src:
                for ss in snrr_index:
                    dat = np.zeros((2, points, 1))
                    for i, typ in enumerate(['im', 're']):
                        path = os.path.join(data_path, 'src_{}_{}.mat'.format(s, typ))
                        dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(s, typ)][ss]
                    for i in range(points / 128):
                        data.append(dat[:, i * 128:(i + 1) * 128, :])
                        label.append(index)

        else:
            for ss in snrr_index:
                dat = np.zeros((2, points, 1))
                for i, typ in enumerate(['im', 're']):
                    path = os.path.join(data_path, 'src_{}_{}.mat'.format(src, typ))
                    dat[i, :, 0] = loadmat(path)['src_{}_{}'.format(src, typ)][ss]

                for i in range(points/128):
                    data.append(dat[:, i * 128:(i + 1) * 128, :])
                    label.append(index)

        print np.array(data).shape
    return np.array(data), np.array(label)


def load_data_from_jpg(data_path, snr_index=None):
    if snr_index or snr_index < 1:
        snrr_index = [snr_index]
    else:
        snrr_index = range(14)
    data = []
    label = []
    for index, src in enumerate(diagram_classes):
        for ss in snrr_index:
            dir_path = os.path.join(data_path, src, 'snr_{}'.format(ss + 1))
            for fil in os.listdir(dir_path):
                img = Image.open(os.path.join(dir_path, fil))
                data.append(np.array(img)/255.0)
                label.append(index)
    # print np.array(data).shape, np.array(label).shape
    return np.array(data), np.array(label)

if __name__ == '__main__':
    load_data_from_jpg('/dataset/RadioML/constellation_diagram/test', snr_index=0)
