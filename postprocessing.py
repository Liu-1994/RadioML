import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    cbar = plt.colorbar(format='%d%%')
    cbar.set_ticks(np.linspace(0, 100, 11))
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=45)
    plt.yticks(num_local, labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_lines(train_his, val_his, saved_name='images.png'):
    x = np.arange(1, len(train_his)+1)
    plt.plot(x, train_his, color='tomato', linewidth=2, label='train')
    plt.plot(x, val_his, color='limegreen', linewidth=2, label='val')
    plt.legend()
    # plt.show()
    plt.savefig(saved_name, format='png', bbox_inches='tight')
    plt.close()