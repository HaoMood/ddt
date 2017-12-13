#!/usr/bin/env python
"""Extract VGG-16 pool5 features for augmented Oxford and Paris dataset.

Features for all images and cropped query images are extracted and saved. One
.npy file for an images.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


from __future__ import division, print_function


__all__ = ['VGGManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-10-03'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-10-03'
__version__ = '1.20'


import argparse
import itertools
import os
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip


import caffe
import numpy as np
import PIL.Image


class VGGManager(object):
    """Extract VGG-16 pool5 features for Oxford and Paris dataset.

    Attributes:
        _paths, dict of (str, str): Image data and feature paths.
        _net: ImageNet pre-trained VGG-16 caffe network.
    """
    def __init__(self, paths):
        self._paths = paths
        self._net = None

    def prepareNet(self, gpu_id):
        """Load VGG-16 model.

        Args:
            gpu_id, int: ordinal of GPU to use.
        """
        print('Prepare VGG-16 ImageNet pre-trained model.')
        self._net = caffe.Net(self._paths['prototxt'], self._paths['model'],
                              caffe.TEST)
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

    def getPool5(self, image_path, pool5_path):
        """Load image and save its pool5 features.

        Args:
            image_path, str: Path to load images.
            pool5_path, str: Path to save pool5 features.
        """
        print('Extract pool5 features.')
        name_list = sorted([
            f for f in os.listdir(self._paths[image_path])
            if os.path.isfile(os.path.join(self._paths[image_path], f))])
        m = len(name_list)
        for i, name_i in enumerate(name_list):
            if i % 100 == 0:
                print('Processing: %d/%d' % (i, m))
            input_list = self._loadImage(os.path.join(self._paths[image_path],
                                                      name_i))
            name_wo_ext_i = os.path.splitext(os.path.basename(name_i))[0]
            name_list = name_wo_ext_i, name_wo_ext_i + '_flip'
            for input_i, save_name_i in zip(input_list, name_list):
                assert input_i.ndim == 3 and input_i.shape[0] == 3

                # Shape for input (data blob is N*D*H*W).
                self._net.blobs['data'].reshape(1, *input_i.shape)
                self._net.blobs['data'].data[...] = input_i
                # Forward pass.
                self._net.forward()
                pool5_i = self._net.blobs['pool5'].data[0]
                assert pool5_i.ndim == 3 and pool5_i.shape[0] == 512

                np.save(os.path.join(self._paths[pool5_path], save_name_i),
                        pool5_i)

    def _loadImage(self, image_full_path):
        """Load the image, normalize to RGB, convert to np.ndarray, and
        preprocess for VGG.

        This is a helper method of getPool5().

        Args:
            image_full_path, str: Full path to an image.

        Return
            input_list, list of np.ndarray of size D*H*W: Prepared VGG input.
        """
        # Load image
        image = PIL.Image.open(image_full_path)
        # TODO(HAO): This is rather convolved, try to simplify it later.
        rgb_image = PIL.Image.new('RGB', image.size)
        rgb_image.paste(image)
        rgb_image_flip = rgb_image.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        # Original image.
        # Image to np.ndarray.
        input_i = np.array(rgb_image, dtype=np.float32)  # HWD format
        input_i = input_i[:, :, ::-1]  # BGR order
        # Subtract mean pixel values of VGG training set.
        input_i -= np.array([104.00698793, 116.66876762, 122.67891434])
        input_i = input_i.transpose((2, 0, 1))  # DHW format

        # Rotate
        # Image to np.ndarray.
        input_i_flip = np.array(rgb_image_flip, dtype=np.float32)  # HWD format
        input_i_flip = input_i_flip[:, :, ::-1]  # BGR order
        # Subtract mean pixel values of VGG training set.
        input_i_flip -= np.array([104.00698793, 116.66876762, 122.67891434])
        input_i_flip = input_i_flip.transpose((2, 0, 1))  # DHW format
        return input_i, input_i_flip


def main():
    """Main function of the program,"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to extract crop query images.')
    parser.add_argument('--gpu', dest='gpu_id', type=int, required=True,
                        help='Ordinal of GPU to use.')
    args = parser.parse_args()
    if args.dataset not in ['oxford', 'paris']:
        raise AttributeError('--dataset parameter must be oxford/paris.')
    if not 0 <= args.gpu_id < 8:
        raise AttributeError('--gpu parameter must be in range [0, 8).')

    image_root = os.path.join('/data/zhangh/data/', args.dataset)
    project_root = '/data/zhangh/project/ddt/'
    data_root = os.path.join('/data/zhangh/project/ddt/data/', args.dataset)
    paths = {
        'image': os.path.join(image_root, 'image/all/'),
        'pool5': os.path.join(data_root, 'pool5/'),
        'prototxt': os.path.join(project_root,
                                 'lib/VGG_ILSVRC_16_pool5.prototxt'),
        'model': os.path.join(project_root,
                              'lib/VGG_ILSVRC_16_layers.caffemodel'),
    }
    for k in paths:
        if k in ['prototxt', 'model']:
            assert os.path.isfile(paths[k])
        else:
            assert os.path.isdir(paths[k])

    manager = VGGManager(paths)
    manager.prepareNet(gpu_id=args.gpu_id)
    manager.getPool5('image', 'pool5')


if __name__ == '__main__':
    main()
