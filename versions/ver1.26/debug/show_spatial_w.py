#!/usr/bin/env python
"""Show DDT spatial weight and CroW spatial weight for randomly selected
images.

Usage:
    ./show_spatial_w.py --dataset oxford

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


from __future__ import division, print_function


__all__ = ['SpatialWViewer']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-29'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-29'
__version__ = '1.6'


import argparse
import itertools
import os
import random
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import sklearn.preprocessing


class SpatialWViewer(object):
    """Show SCPP spatial weight and CroW spatial weight for randomly selected
    images.

    Attributes:
        _paths, dict of (str, str): Feature paths.
    """
    def __init__(self, paths):
        self._paths = paths

    def generateRandom(self):
        """Select a random image filename from oxford dataset.

        Return:
            image_name, str: Randomly selected image name without extension.
        """
        all_names = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths['all_image'])
            if os.path.isfile(self._paths['all_image'] + f)])
        # random.randrange(i, j) returns a random integer in range [i, j).
        # Select a random image.
        i = random.randrange(0, len(all_names))
        return all_names[i]

    def view(self, image_name):
        """Show the SCPP spatial weight and CroW spatial weight for the given
        image.

        Args:
            image_name, str: Randomly selected image name without extension.
        """
        # Load image
        image = PIL.Image.open('%s.jpg' % os.path.join(
            self._paths['all_image'], image_name))
        plt.figure()
        plt.imshow(image)
        plt.title('Image: %s' % image_name)

        # Load pool5 feature.
        pool5_i = np.load('%s.npy' % os.path.join(self._paths['all_pool5'],
                                                  image_name))
        D, H, W = pool5_i.shape
        assert D == 512

        # DDT spatial weight.
        pc_i = np.load('%s.npy' % os.path.join(self._paths['all_pc1'],
                                                image_name))
        pc_i = np.maximum(pc_i, 0)
        assert pc_i.shape == (H * W, 2)

        pc1_i = pc_i[:, 0:1]
        assert pc1_i.shape == (H * W, 1)
        if np.sum(pc1_i) <= 1e-3:
            print("%s's DDT weight is 0." % image_name)
            ddt_spatial_w = np.zeros((H, W))
        else:
            # l2,sqrt-normalize.
            ddt_spatial_w = sklearn.preprocessing.normalize(pc1_i.T)
            ddt_spatial_w = np.sqrt(ddt_spatial_w)
            ddt_spatial_w = np.reshape(ddt_spatial_w, (H, W))
        plt.figure()
        plt.imshow(ddt_spatial_w)
        plt.colorbar()
        plt.title('DDT spatial weight PC1')

        pc2_i = pc_i[:, 1:2]
        assert pc1_i.shape == (H * W, 1)
        if np.sum(pc2_i) <= 1e-3:
            print("%s's DDT weight is 0." % image_name)
            ddt_spatial_w = np.zeros((H, W))
        else:
            # l2,sqrt-normalize.
            ddt_spatial_w = sklearn.preprocessing.normalize(pc2_i.T)
            ddt_spatial_w = np.sqrt(ddt_spatial_w)
            ddt_spatial_w = np.reshape(ddt_spatial_w, (H, W))
        plt.figure()
        plt.imshow(ddt_spatial_w)
        plt.colorbar()
        plt.title('DDT spatial weight PC2')

        # CroW spatial weight.
        S = np.sum(pool5_i, axis=0, keepdims=True)
        assert S.shape == (1, H, W)
        if np.sum(S) == 0:
            crow_spatial_w = np.ones((H, W))
        else:
            crow_spatial_w = np.sqrt(sklearn.preprocessing.normalize(
                np.reshape(S, (1, H * W))))
            crow_spatial_w = np.reshape(crow_spatial_w, (H, W))
        plt.figure()
        plt.imshow(crow_spatial_w)
        plt.colorbar()
        plt.title('CroW spatial weight')
        plt.show()


def main():
    """Main function of this program."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to view spatial weights.')
    args = parser.parse_args()
    if args.dataset not in ['oxford', 'paris']:
        raise AttributeError('--dataset parameter must be oxford/paris.')

    image_root = os.path.join('/data/zhangh/data/', args.dataset)
    data_root = os.path.join('/data/zhangh/project/ddt/data/', args.dataset)
    paths = {
        'all_image': os.path.join(image_root, 'image/all/'),
        'all_pool5': os.path.join(image_root, 'pool5/all/'),
        'all_pc1': os.path.join(data_root, 'pc1-2/all/'),
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    viewer = SpatialWViewer(paths)
    image_name = viewer.generateRandom()
    # image_name = 'oxford_002701'
    viewer.view(image_name)


if __name__ == '__main__':
    main()
