import sys
import argparse
from models import *
from preprocessing import *
from sklearn.metrics import confusion_matrix
from postprocessing import *



def run(args):

    if not args.mode in ['predict', 'evaluate']:
        print "mode error, it should in ['predict', 'evaluate'], now it is {}".format(args.mode)
        exit()

    snr_index = (int(args.saved_model.split('/')[-2].split('_')[-2]) + 8) / 2
    model = keras.models.load_model(args.saved_model)

    num_class = len(radio_classes)

    if args.mode == 'evaluate':
        train_data, train_label, test_data, test_label = load_data_from_mat(args.data_path, snr_index)
        test_label = keras.utils.to_categorical(test_label, num_classes=num_class)
        rel = model.evaluate(test_data, test_label, batch_size=args.batch_size, verbose=2)
        print 'loss: {:.4f}'.format(rel[0])
        print 'accuracy: {:.4f}'.format(rel[1])
        if args.plot_confusion_matrix:
            pred_y = model.predict(test_data, batch_size=args.batch_size, verbose=2)
            cm = confusion_matrix(np.argmax(test_label, axis=1), np.argmax(pred_y, axis=1))
            print cm
            label_name = []
            for rad in radio_classes:
                if isinstance(rad, list):
                    label_name.append('qams')
                else:
                    label_name.append(rad)
            plot_confusion_matrix(cm, label_name, 'Confusion_matrix')
            plt.savefig('{}_dB_confusion_matrix.png'.format(args.saved_model.split('/')[-2].split('_')[-2]), format='png', bbox_inches='tight')
    else:
        train_data, train_label, test_data, test_label = load_data_from_mat(args.data_path, snr_index)
        results = model.predict(test_data, batch_size=args.batch_size, verbose=2)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/signal_dataset')
    parser.add_argument('--saved_model', type=str, default='checkpoints/signal_to_noise_ratio_0_dB/weights_200.hdf5')
    parser.add_argument('--mode', type=str, default='evaluate', help="mode is 'evaluate' or 'predict'.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--plot_confusion_matrix', type=bool, default=True)

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))