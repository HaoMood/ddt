#!/usr/bin/env python
"""Extract VGG convolution feature maps for Oxford5k and Paris6k datasets.

Features for all database images and cropped query images are extracted and
saved. One .npy file for an image.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


import os

import caffe
import numpy as np
import PIL.Image


__all__ = ['VGGManager']
__author__ = 'Hao Zhang'
__copyright__ = '2017 LAMDA'
__date__ = '2017-10-27'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Production'
__updated__ = '2017-11-24'
__version__ = '2.0'


class VGGManager(object):
    """Extract VGG convolution feature maps for Oxford5k and Paris6k dataset.

    Attributes:
        _net: ImageNet pre-trained VGG caffe model.
        _paths, dict of (str, str): Image data and feature paths.
    """
    def __init__(self, paths):
        print('Initiate.')
        self._paths = paths
        self._net = None

    def prepareNet(self, gpu_id):
        """Load VGG model.

        Args:
            gpu_id, int: ordinal of GPU to use.
        """
        print('Prepare VGG ImageNet pre-trained model.')
        self._net = caffe.Net(self._paths['prototxt'], self._paths['model'],
                              caffe.TEST)
        caffe.set_device(gpu_id)
        caffe.set_mode_gpu()

    def getConv(self, image_path, conv_path):
        """Load image and save its convolution feature maps.

        Args:
            image_path, str: Path to load images.
            conv_path, str: Path to save conv features.
        """
        print('Extract conv features.')
        name_list = sorted([
            f for f in os.listdir(self._paths[image_path])
            if os.path.isfile(os.path.join(self._paths[image_path], f))])
        m = len(name_list)
        for i, name_i in enumerate(name_list):
            if i % 200 == 0:
                print('Processing: %d/%d' % (i, m))
            # Load input image and convert to np.ndarray.
            input_i = VGGManager._loadImage(os.path.join(
                self._paths[image_path], name_i))
            assert input_i.ndim == 3 and input_i.shape[0] == 3

            # Shape for input (data blob is N*D*H*W).
            self._net.blobs['data'].reshape(1, *input_i.shape)
            self._net.blobs['data'].data[...] = input_i

            # Forward pass.
            self._net.forward()
            conv_i = self._net.blobs['conv5_4'].data[0]
            assert conv_i.ndim == 3 and conv_i.shape[0] == 512

            # Save onto disk.
            name_wo_ext_i = os.path.splitext(os.path.basename(name_i))[0]
            np.save(os.path.join(self._paths[conv_path], name_wo_ext_i),
                    conv_i)

    def _loadImage(image_full_path):
        """Load the image, normalize to RGB, convert to np.ndarray, and
        preprocess for VGG.

        This is a helper function of getconv().

        Args:
            image_full_path, str: Full path to an image.

        Return
            input_i, np.ndarray of size D*H*W: Prepared VGG input.
        """
        # Load image
        image = PIL.Image.open(image_full_path)
        rgb_image = PIL.Image.new('RGB', image.size)
        rgb_image.paste(image)

        # Image to np.ndarray
        input_i = np.array(rgb_image, dtype=np.float32)  # HWD format
        input_i = input_i[:, :, ::-1]  # BGR order

        # Subtract mean pixel values of VGG training set.
        input_i -= np.array([104.00698793, 116.66876762, 122.67891434])
        input_i = input_i.transpose((2, 0, 1))  # DHW format
        return input_i


def main():
    """Main function of the program,"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to extract crop query images.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model to extract features.')
    parser.add_argument('--gpu', dest='gpu_id', type=int, required=True,
                        help='Ordinal of GPU to use.')
    args = parser.parse_args()
    if args.dataset not in ['oxford5k', 'paris6k']:
        raise AttributeError('--dataset parameter must be oxford5k/paris6k.')
    if args.model not in ['vgg16', 'vgg19']:
        raise AttributeError('--model parameter must be vgg16/vgg19.')
    if args.gpu_id < 0:
        raise AttributeError('--gpu parameter must be >=0.')

    project_root = os.popen('pwd').read().strip()
    data_root = os.path.join(os.path.join(project_root, 'data'), args.dataset)
    paths = {
        'all_image': os.path.join(data_root, 'image/all/'),
        'crop_image': os.path.join(data_root, 'image/crop/'),
        'all_conv': os.path.join(os.path.join(os.path.join(
            data_root, 'conv'), args.model), 'all/'),
        'crop_conv': os.path.join(os.path.join(os.path.join(
            data_root, 'conv'), args.model), 'crop/'),
        'prototxt': os.path.join(project_root,
                                 ('lib/VGG_ILSVRC_16_pool5.prototxt'
                                  if args.model == 'vgg16'
                                  else 'lib/VGG_ILSVRC_19_pool5.prototxt')),
        'model': os.path.join(project_root,
                              ('lib/VGG_ILSVRC_16_layers.caffemodel'
                               if args.model == 'vgg16'
                               else 'lib/VGG_ILSVRC_19_layers.caffemodel')),
    }
    for k in paths:
        if k in ['prototxt', 'model']:
            assert os.path.isfile(paths[k])
        else:
            assert os.path.isdir(paths[k])

    manager = VGGManager(paths)
    manager.prepareNet(args.gpu_id)
    manager.getConv('all_image', 'all_conv')
    manager.getConv('crop_image', 'crop_conv')


if __name__ == '__main__':
    main()
