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
