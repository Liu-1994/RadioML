import sys
import random
import argparse
from models import *
from keras.optimizers import SGD
from preprocessing import *
from postprocessing import *


def run(args):

    snr_index = (args.signal_to_noise_ratio + 8) / 2

    if args.signal_to_noise_ratio > 18 or snr_index < 0 or args.signal_to_noise_ratio % 2 != 0:
        print 'Error!!! signal_to_noise_ratio should in -8:2:18, now it is {}'.format(args.signal_to_noise_ratio)
        exit()
    if not args.mode in ['raw', 'diagram']:
        print 'Error!!! args.mode should be "raw" or "diagram", now it is {}'.format(args.mode)

    if args.mode == 'raw':

        callbacks = []
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.log_path, 'signal_to_noise_ratio_{}_dB'.format(str(args.signal_to_noise_ratio))),
            histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)

        checkpoint_path = os.path.join(args.checkpoint_path, 'signal_to_noise_ratio_{}_dB'.format(str(args.signal_to_noise_ratio)))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, 'weights_{epoch:02d}.hdf5'), period=20,
            save_weights_only=False, save_best_only=False)
        callbacks.append(checkpoint)

        num_class = len(radio_classes)
        print 'the number of signal classes: ', num_class

        # train_data, train_label, val_data, val_label = load_train_val_data_from_mat(args.data_path)
        train_data, train_label = load_data_from_mat(os.path.join(args.data_path, 'train'), snr_index=snr_index)
        val_data, val_label = load_data_from_mat(os.path.join(args.data_path, 'test'), snr_index=snr_index)
        train_label = keras.utils.to_categorical(train_label, num_classes=num_class)
        val_label = keras.utils.to_categorical(val_label, num_classes=num_class)
        shuffle = np.arange(train_data.shape[0])
        random.shuffle(shuffle)
        train_data = train_data[shuffle]
        train_label = train_label[shuffle]

        print 'train data: ', train_data.shape, 'train label: ', train_label.shape
        print 'val data: ', val_data.shape, 'val label: ', val_label.shape

        model = dr_cnn(num_class)
        sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        result = model.fit(train_data, train_label,
                           validation_data=[val_data, val_label],
                           batch_size=args.batch_size, epochs=args.nb_epoch, verbose=2, shuffle=True,
                           callbacks=callbacks)
        plot_lines(result.history['acc'], result.history['val_acc'],
                   saved_name=os.path.join(args.log_path, 'signal_to_noise_ratio_{}_dB'.format(
                       str(args.signal_to_noise_ratio)), 'acc.png'))
        plot_lines(result.history['loss'], result.history['val_loss'],
                   saved_name=os.path.join(args.log_path, 'signal_to_noise_ratio_{}_dB'.format(
                       str(args.signal_to_noise_ratio)), 'loss.png'))

    else:
        callbacks = []
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.log_path, 'constellation_diagram', 'signal_to_noise_ratio_{}_dB'.format(str(args.signal_to_noise_ratio))),
            histogram_freq=0, write_graph=True)
        callbacks.append(tensorboard)

        args.log_path = os.path.join(args.log_path, 'constellation_diagram')
        args.checkpoint_path = os.path.join(args.checkpoint_path, 'constellation_diagram')

        checkpoint_path = os.path.join(args.checkpoint_path,
                                       'signal_to_noise_ratio_{}_dB'.format(str(args.signal_to_noise_ratio)))
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_path, 'weights_{epoch:02d}.hdf5'), period=20,
            save_weights_only=False, save_best_only=False)
        callbacks.append(checkpoint)

        num_class = len(diagram_classes)
        print 'the number of diagram signal classes: ', num_class

        train_data, train_label = load_data_from_jpg(os.path.join(args.data_path, 'train'), snr_index=snr_index)
        val_data, val_label = load_data_from_jpg(os.path.join(args.data_path, 'test'), snr_index=snr_index)
        train_label = keras.utils.to_categorical(train_label, num_classes=num_class)
        val_label = keras.utils.to_categorical(val_label, num_classes=num_class)
        shuffle = np.arange(train_data.shape[0])
        random.shuffle(shuffle)
        train_data = train_data[shuffle]
        train_label = train_label[shuffle]

        print 'train data: ', train_data.shape, 'train label: ', train_label.shape
        print 'val data: ', val_data.shape, 'val label: ', val_label.shape

        model = latter_cnn(num_class)
        sgd = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['acc'])

        result = model.fit(train_data, train_label,
                           validation_data=[val_data, val_label],
                           batch_size=args.batch_size, epochs=args.nb_epoch, verbose=2, shuffle=True,
                           callbacks=callbacks)
        plot_lines(result.history['acc'], result.history['val_acc'],
                   saved_name=os.path.join(args.log_path, 'signal_to_noise_ratio_{}_dB'.format(
                       str(args.signal_to_noise_ratio)), 'acc.png'))
        plot_lines(result.history['loss'], result.history['val_loss'],
                   saved_name=os.path.join(args.log_path, 'signal_to_noise_ratio_{}_dB'.format(
                       str(args.signal_to_noise_ratio)), 'loss.png'))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/dataset/RadioML/constellation_diagram')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints')
    parser.add_argument('--nb_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--signal_to_noise_ratio', type=int, default=18)
    parser.add_argument('--mode', type=str, default='diagram', help="'diagram' or 'raw")

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))