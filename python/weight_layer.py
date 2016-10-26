import caffe
import numpy as np
import scipy as sc


class WeightLayer(caffe.Layer):
    """
    Weight layer.
    """

    def setup(self, bottom, top):
        self.gt_labels = [0, 1, 2]

    def forward(self, bottom, top):
        labels = bottom[0].data
        weights = np.zeros(labels.shape)
        for idx, label in enumerate(labels):
            for gt_label in self.gt_labels:
                mask = label == gt_label
                count = mask.sum()
                if count != 0:
                    weights[idx][mask] = 1. / count

        top[0].data[...] = weights / labels.shape[0]
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        top[0].reshape(*bottom[0].data.shape)
        pass
