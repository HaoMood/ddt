#!/usr/bin/env python
"""Compute DDT PC1 projections."""


import os
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import sklearn.decomposition


__all__ = ['DDTManager']
__author__ = 'Hao Zhang'
__copyright__ = 'Copyright @2017 LAMDA'
__date__ = '2017-09-29'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-11-24'
__version__ = '2.0'


class DDTManager(object):
    """Manager class to perform PCA on deep descriptors.

    Attributes:
        _paths, dict of (str, str): Feature paths.
    """
    def __init__(self, paths):
        self._paths = paths

    def fit(self):
        """Load all deep descriptors and perform PCA."""
        print('Load all deep descriptors.')
        all_descriptors = self._loadFeature('all_conv')

        print('Perform PCA on those deep descriptors.')
        pca_model = sklearn.decomposition.PCA(n_components=1)
        pca_model.fit(np.vstack(all_descriptors))
        full_path = os.path.join(self._paths['pca_model'],
                                 'descriptor_model.pkl')
        pickle.dump(pca_model, open(full_path, 'w'))

    def _loadFeature(self, conv_path):
        """Load all deep descriptors.

        This is a helper method of fit().

        Args:
            conv_path, str: Path of .npy conv features.

        Return:
            descriptor_list, list of np.ndarray of size HW*D.
        """
        name_list = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths[conv_path])
            if os.path.isfile(os.path.join(self._paths[conv_path], f))])
        m = len(name_list)
        descriptor_list = []
        for i, name_i in enumerate(name_list):
            if i % 200 == 0:
                print('Processing %d/%d' % (i, m))
            # Load conv feature.
            conv_i = np.load('%s.npy' %
                             os.path.join(self._paths[conv_path], name_i))
            D, H, W = conv_i.shape
            assert D == 512

            # Reshape into size HW*D.
            conv_i = np.reshape(conv_i, (D, H * W)).T
            descriptor_list.append(conv_i)
        return descriptor_list

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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset for DDT.')
    parser.add_argument('--model', dest='model', type=str, required=True,
                        help='Model to extract features.')
    args = parser.parse_args()
    if args.dataset not in ['oxford5k', 'paris6k']:
        raise AttributeError('--dataset parameter must be oxford5k/paris6k.')
    if args.model not in ['vgg16', 'vgg19']:
        raise AttributeError('--model parameter must be vgg16/vgg19.')

    project_root = os.popen('pwd').read().strip()
    data_root = os.path.join(os.path.join(project_root, 'data'), args.dataset)
    paths = {
        'all_conv': os.path.join(os.path.join(os.path.join(
            data_root, 'conv'), args.model), 'all/'),
        'pca_model': data_root,
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    ddt_manager = DDTManager(paths)
    ddt_manager.fit()


if __name__ == '__main__':
    main()
