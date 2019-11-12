import random
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import matplotlib.cm as mpcm


def plot_classification_data(images, cls_true, cls_pred=None, class_names=None, save_name=None):
    fig, axes = plt.subplots(3, 3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")
        if class_names is not None:
            cls_true_name = class_names[cls_true[i]]
        else:
            cls_true_name = cls_true[i]
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true_name)
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])
        ax.set_xlabel(xlabel)
        # # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def plot_semantics_data(images, lables, cls_pred=None, class_names=None, save_name=None):
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(8, 2)
    # fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        if i % 2 == 0:
            ax.imshow(images[i // 2], )
        else:
            ax.imshow(lables[i // 2], )

        # # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def plot_detection_data1(img, classes, scores, bboxes, figsize=(10, 10), linewidth=1.5, save_name=None):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())
            ymin = int(bboxes[i, 0] * height)
            xmin = int(bboxes[i, 1] * width)
            ymax = int(bboxes[i, 2] * height)
            xmax = int(bboxes[i, 3] * width)
            rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                 ymax - ymin, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = str(cls_id)
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=12, color='white')
    if save_name != None:
        plt.savefig(save_name)
    plt.show()


def plot_detection_data(img, classes, scores, bboxes, class_names, figsize=(10, 10), linewidth=3, save_name=None):
    """Visualize bounding boxes. Largely inspired by SSD-MXNET!
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    colors = dict()
    for i in range(classes.shape[0]):
        cls_id = int(classes[i])
        if cls_id >= 0:
            score = scores[i]
            if cls_id not in colors:
                colors[cls_id] = (random.random(), random.random(), random.random())

            xmin = bboxes[i, 0]
            ymin = bboxes[i, 1]
            w = (bboxes[i, 2] - bboxes[i, 0])
            h = (bboxes[i, 3] - bboxes[i, 1])
            rect = plt.Rectangle((xmin, ymin), w,
                                 h, fill=False,
                                 edgecolor=colors[cls_id],
                                 linewidth=linewidth)
            plt.gca().add_patch(rect)
            class_name = class_names[cls_id]
            plt.gca().text(xmin, ymin - 2,
                           '{:s} | {:.3f}'.format(class_name, score),
                           bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                           fontsize=15, color='white')
    if save_name != None:
        plt.axis('off')
        plt.savefig(save_name, bbox_inches='tight')
    plt.show()
