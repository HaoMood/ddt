#!/usr/bin/env python
"""Compute DDT PC1 projections."""


from __future__ import division, print_function


__all__ = ['DDTManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-29'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-09-29'
__version__ = '1.0'


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

import numpy as np
import sklearn.decomposition


class DDTManager(object):
    """Manager class for split Oxford images, perform PCA on each cluster,
    and compute PC1 projection on Oxford features.

    Attributes:
        _paths, dict of (str, str): Data and project paths.
        _pca_model: PCA models for DDT.
    """
    def __init__(self, paths):
        self._paths = paths
        self._pca_model = None

    def fit(self):
        """Load via pool5 features and fit a PCA model."""
        print('Load via pool5 features.')
        pool5_list = sorted([
            f for f in os.listdir(self._paths['via_all_pool5'])
            if os.path.isfile(os.path.join(self._paths['via_all_pool5'], f))])
        m = len(pool5_list)
        all_pool5 = []
        for i, f in enumerate(pool5_list):
            if i % 100 == 0:
                print('procesing %d/%d.' % (i, m))
            pool5_i = np.load(os.path.join(self._paths['via_all_pool5'], f))
            D, H, W = pool5_i.shape
            assert D == 512
            pool5_i = np.reshape(pool5_i, (D, H * W)).T
            all_pool5.append(pool5_i)
        all_pool5 = np.vstack(all_pool5)
        assert all_pool5.shape[1] == 512

        print('Fit PCA.')
        self._pca_model = sklearn.decomposition.PCA(n_components=1)
        self._pca_model.fit(all_pool5)

    def apply(self, pool5_path, pc1_path):
        """Project test pool5 features according to the PCA model, for all
        images/cropped query images, respectively.

        Args:
            pool5_path, str: Pool5 feature path.
            pc1_path, str: PC1 projection path.
        """
        print('Compute PC1 projection for %s.' % pool5_path)
        pool5_names = sorted([
            f for f in os.listdir(self._paths[pool5_path])
            if os.path.isfile(os.path.join(self._paths[pool5_path], f))])
        m = len(pool5_names)
        for i, f in enumerate(pool5_names):
            if i % 100 == 0:
                print('Processing %d/%d' % (i, m))
            pool5_i = np.load(os.path.join(self._paths[pool5_path], f))
            D, H, W = pool5_i.shape
            assert D == 512
            pool5_i = np.reshape(pool5_i, (D, H * W)).T
            pc1_i = self._pca_model.transform(pool5_i)
            np.save(os.path.join(self._paths[pc1_path],
                                 os.path.splitext(os.path.basename(f))[0]),
                    pc1_i)


def main():
    """Main function of the program."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, required=True,
                        help='Dataset to evaluate.')
    parser.add_argument('--via', dest='via', type=str, required=True,
                        help='Dataset to assistant PCA whitening.')
    args = parser.parse_args()
    if args.test not in ['oxford', 'paris']:
        raise AttributeError('--test parameter must be oxford/paris.')
    if args.via not in ['oxford', 'paris']:
        raise AttributeError('--via parameter must be oxford/paris.')

    test_image_root = os.path.join('/data/zhangh/data/', args.test)
    via_image_root = os.path.join('/data/zhangh/data/', args.via)
    test_data_root = os.path.join('/data/zhangh/project/scpp/data/', args.test)
    paths = {
        'test_all_pool5': os.path.join(test_image_root, 'pool5/all/'),
        'test_crop_pool5': os.path.join(test_image_root, 'pool5/crop/'),
        'test_full_pool5': os.path.join(test_image_root, 'pool5/full/'),
        'via_all_pool5': os.path.join(via_image_root, 'pool5/all/'),
        'test_all_pc1': os.path.join(test_data_root, 'pc1/all/'),
        'test_crop_pc1': os.path.join(test_data_root, 'pc1/crop/'),
        'test_full_pc1': os.path.join(test_data_root, 'pc1/full/'),
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    ddt_manager = DDTManager(paths)
    ddt_manager.fit()
    ddt_manager.apply('test_all_pool5', 'test_all_pc1')
    ddt_manager.apply('test_crop_pool5', 'test_crop_pc1')
    ddt_manager.apply('test_full_pool5', 'test_full_pc1')


if __name__ == '__main__':
    main()
