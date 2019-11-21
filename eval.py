import sys
import argparse
from models import *
from preprocessing import *
from sklearn.metrics import confusion_matrix
from postprocessing import *


def run(args):

    if not args.type in ['predict', 'evaluate']:
        print "type error, it should in ['predict', 'evaluate'], now it is {}".format(args.type)
        exit()

    snr_index = (int(args.saved_model.split('/')[-2].split('_')[-2]) + 8) / 2
    model = keras.models.load_model(args.saved_model)

    if args.type == 'evaluate':
        print args.saved_model
        if model.input_shape == (None, 2, 128, 1):
            num_class = len(radio_classes)
            print 'the number of raw IQ signal classes: ', num_class
            test_data, test_label = load_data_from_mat(os.path.join(args.data_path, 'test'), snr_index=snr_index)
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
            num_class = len(diagram_classes)
            print 'the number of diagram signal classes: ', num_class
            test_data, test_label = load_data_from_jpg(os.path.join(args.data_path, 'test'), snr_index=snr_index)
            test_label = keras.utils.to_categorical(test_label, num_classes=num_class)
            rel = model.evaluate(test_data, test_label, batch_size=args.batch_size, verbose=2)
            print 'loss: {:.4f}'.format(rel[0])
            print 'accuracy: {:.4f}'.format(rel[1])
    else:
        print args.saved_model

        if model.input_shape == (None, 2, 128, 1):
            test_data, test_label = load_data_from_mat(os.path.join(args.data_path, 'test'), snr_index=snr_index)
        else:
            test_data, test_label = load_data_from_jpg(os.path.join(args.data_path, 'test'), snr_index=snr_index)

        results = model.predict(test_data, batch_size=args.batch_size, verbose=2)
        results_arg = np.argmax(results, axis=1)
        for ind, rel in enumerate(results_arg):
            print 'the {} th, real: {}, predict: {}, prob: {:0.4f}'.format(
                ind+1, diagram_classes[test_label[ind]], diagram_classes[rel], results[ind, rel])


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/dataset/RadioML/constellation_diagram')
    parser.add_argument('--saved_model', type=str, default='checkpoints/constellation_diagram/signal_to_noise_ratio_18_dB/weights_200.hdf5')
    parser.add_argument('--type', type=str, default='predict', help="mode is 'evaluate' or 'predict'.")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--plot_confusion_matrix', type=bool, default=True)

    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    run(parse_arguments(sys.argv[1:]))