import os
import caffe
import numpy as np
import numpy.random as rnd
from PIL import Image

class LSPDataLayer(caffe.Layer):
    """
    Load (input image, label image) pairs from the LSP dataset,
    one-at-a-time while reshaping the net to preserve dimensions.
    Use this to feed data to the fully convolutional network.
    """

    def setup(self, bottom, top):
        """
        Initialize the data layer from given parameters
        
        dir: base dir of images, labels, and input files
        input: file with the names of the images
        label: file with the names of the label files
        mean: mean value to subtract from images
        randomize: randomize the training/test set
        seed: seed for randomization
        
        """
        
        # configure the data layer
        params = eval(self.param_str)
        self.dir = params['dir']
        self.image_file = params['input']
        self.label_file = params['label']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        
        # load images and label names, both need to have the right order
        self.images = open(os.path.join(self.dir, self.image_file), 'r').read().splitlines()
        self.labels = open(os.path.join(self.dir, self.label_file), 'r').read().splitlines()
        self.idx = 0
        
        # if randomization desired, just permute the indices to ensure
        # every instance is visited at least once (per epoch)
        if self.random:
            self.indices = rnd.permutation(len(self.images))
        else:
            self.indices = np.arange(len(self.images))
    
    def reshape(self, bottom, top):
        # load image and label
        self.data = self.load_image(self.images[self.indices[self.idx]])
        self.label = self.load_label(self.labels[self.indices[self.idx]])
        
        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)
    
    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label
    
        if self.idx + 1 == len(self.images):
            self.idx = 0
            if self.random:
                # one epoch has passed -> reshuffle
                self.indices = rnd.permutation(len(self.images))
        else:
            self.idx += 1

    def backward(self, top, propagate_down, bottom):
        # nothing to do here
        pass
    
    def load_image(self, name):
        img = Image.open(name)
        img = np.array(img, dtype=np.float32)
        img = img - self.mean
        img = img[:, :, ::-1]
        img = img.transpose((2, 0, 1))
        
        SCALES = 5
        for i in xrange(SCALES): 
            if i == 0:
                firstScale = img
            else:
                firstScale = np.concatenate((firstScale, img), axis = 0)
    
        img = firstScale
        
        return img
    
    def load_label(self, name):
        label = Image.open(name)
        label = np.array(label, dtype=np.uint8)
        label = label[np.newaxis, ...]
        return label


