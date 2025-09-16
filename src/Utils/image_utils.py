import matplotlib.pyplot as plt
import numpy as np
import itertools, math



# FROM https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def image_grid(images, preds, class_names):
    n = images.shape[0]
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))

    imgs = images.permute(0, 2, 3, 1)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i >= n:
            ax.axis('off')
            continue
        ax.set_axis_off()

        ax.imshow(imgs[i].numpy(), vmin=0, vmax=1)
        title = class_names[preds[i]]
        ax.set_title(title, fontsize=15, fontweight='bold', pad=2)

    return fig




    