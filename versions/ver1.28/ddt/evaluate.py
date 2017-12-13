#!/usr/bin/env python
"""Find features for query images and evaluate the result.

We use the DDT PC1 projections as the part detector to perform weighted
sum-pooling of pool5 features to obtain the final descriptor.

This program is modified from CroW:
    https://github.com/yahoo/crow/

The compute_ap executable file is modified from VGG:
    http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/
To compute the average precision for a ranked list of query christ_church_1,
run:
    ./compute_ap christ_church_1 ranked_list.txt
"""


from __future__ import division, print_function


__all__ = ['EvaluateManager']
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
import sys
if sys.version[0] == '2':
    filter = itertools.ifilter
    input = raw_input
    map = itertools.imap
    range = xrange
    zip = itertools.izip
import tempfile

import numpy as np
import sklearn.decomposition
import sklearn.neighbors
import sklearn.preprocessing


class EvaluateManager(object):
    """Manager class to compute mAP.

    Attributes:
        _paths, dict of (str, str): Feature paths.
        _pca_model: sklearn.decomposition.PCA model for final PCA whitening.
    """
    def __init__(self, paths):
        self._paths = paths
        self._pca_model = None

    def fitPca(self, dim, whiten):
        """Compute PCA whitening paramters from the via dataset.

        Args:
            via_pool5_path, str: Pool5 feature path of the via dataset.
            dim, int: Dimension of the final retrieval descriptor.
            whiten, bool: Whether to use whitening.
        """
        print('Load via dataset features.')
        via_all_descriptors, _ = self._loadFeature(
            'via_all_pc1', 'via_all_pool5')
        via_all_descriptors = np.vstack(via_all_descriptors)
        assert via_all_descriptors.shape[1] == 512

        print('Fit PCA whitening paramters from via dataset.')
        sklearn.preprocessing.normalize(via_all_descriptors, copy=False)
        self._pca_model = sklearn.decomposition.PCA(n_components=dim,
                                                    whiten=whiten)
        self._pca_model.fit(via_all_descriptors)

    def evaluate(self, crop):
        """Evaluate the retrieval results.

        Args:
            crop, bool: Whether to use whitening.
        """
        print('Load test dataset features.')
        test_all_descriptors, test_all_names = self._loadFeature(
            'test_all_pc1', 'test_all_pool5')
        test_query_pc1 = 'test_crop_pc1' if crop else 'test_full_pc1'
        test_query_pool5 = 'test_crop_pool5' if crop else 'test_full_pool5'
        test_crop_descriptors, test_crop_names = self._loadFeature(
            test_query_pc1, test_query_pool5)
        test_all_descriptors, test_crop_descriptors = self._normalization(
            test_all_descriptors, test_crop_descriptors)
        assert test_all_descriptors.shape[1] == test_crop_descriptors.shape[1]

        # Iterate queries, process them, rank results, and evaluate mAP.
        all_ap = []
        for i in xrange(len(test_crop_names)):
            knn_model = sklearn.neighbors.NearestNeighbors(
                n_neighbors=len(test_all_descriptors))
            knn_model.fit(test_all_descriptors)
            _, ind = knn_model.kneighbors(
                test_crop_descriptors[i].reshape(1, -1))
            ap = self._getAP(ind[0], test_crop_names[i], test_all_names)
            all_ap.append(ap)
        m_ap = np.mean(np.array(all_ap))
        print('mAP is', m_ap)

    def _loadFeature(self, pc1_path, pool5_path):
        """Load and process pool5 features into weighted-pooled descriptors.

        This is a helper function of fitPca() and evaluate().

        Args:
            pc1_path, str: Path of PC1 projections.
            pool5_path, str: Path of .npy pool5 features.

        Return:
            descriptor_list, list of np.ndarray of size 1*512: List of
                weighted-pooled descriptors.
            name_list, list of str: List of image names without extensions.
        """
        name_list = sorted([
            os.path.splitext(os.path.basename(f))[0]
            for f in os.listdir(self._paths[pool5_path])
            if os.path.isfile(os.path.join(self._paths[pool5_path], f))])
        m = len(name_list)
        descriptor_list = []
        for i, name_i in enumerate(name_list):
            if i % 100 == 0:
                print('Processing %d/%d' % (i, m))
            # Load pool5 feature.
            pool5_i = np.load('%s.npy' % os.path.join(self._paths[pool5_path],
                                                      name_i))
            D, H, W = pool5_i.shape
            assert D == 512

            # DDT spatial weight.
            pc1_i = np.load('%s.npy' % os.path.join(self._paths[pc1_path],
                                                    name_i))
            assert pc1_i.shape == (H * W, 1)
            pc1_i = np.maximum(pc1_i, 0)
            if np.sum(pc1_i) <= 1e-3:
                print("%s's DDT weight is 0." % name_i)
                ddt_spatial_w = np.zeros((1, H, W))
            else:
                ddt_spatial_w = sklearn.preprocessing.normalize(pc1_i.T)
                ddt_spatial_w = np.sqrt(ddt_spatial_w)
                ddt_spatial_w = np.reshape(ddt_spatial_w, (1, H, W))

            # CroW channel weight.
            Q = np.sum(pool5_i > 0, axis=(1, 2), keepdims=True) / (H * W)
            assert Q.shape == (512, 1, 1)
            crow_channel_w = np.zeros((512, 1, 1))
            for d in range(Q.shape[0]):
                if Q[d, 0, 0] > 0:
                    crow_channel_w[d, 0, 0] = np.log(np.sum(Q) / Q[d, 0, 0])

            # Weighted-pooled descriptors.
            # DDT only.
            # descriptor_i = np.sum(ddt_spatial_w * pool5_i,
            #                       axis=(1, 2))[np.newaxis, :]
            # DDT + channel weighting.
            descriptor_i = np.sum(ddt_spatial_w * (crow_channel_w * pool5_i),
                                 axis=(1, 2))[np.newaxis, :]
            assert descriptor_i.shape == (1, 512)
            descriptor_list.append(descriptor_i)
        return descriptor_list, name_list

    def _normalization(self, test_all_descriptors, test_crop_descriptors):
        """l2 normalize, PCA whitening, and l2 normalize again for test all and
        cropped query descriptors.

        This is a helper function of evaluate().

        Args:
            test_all_descriptors, np.ndarray of m*512: Before normalize.
            test_crop_descriptors, np.ndarray of m*512: Before normalize.

        Return
            test_all_descriptors, np.ndarray of m*512: After normalize.
            test_crop_descriptors, np.ndarray of m*512: After normalize.
        """
        test_all_descriptors = np.vstack(test_all_descriptors)
        test_crop_descriptors = np.vstack(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)

        test_all_descriptors = self._pca_model.transform(test_all_descriptors)
        test_crop_descriptors = self._pca_model.transform(test_crop_descriptors)

        sklearn.preprocessing.normalize(test_all_descriptors, copy=False)
        sklearn.preprocessing.normalize(test_crop_descriptors, copy=False)
        return test_all_descriptors, test_crop_descriptors

    def _getAP(self, ind, query_name, all_names):
        """Given a query, compute average precision for the results by calling
        to the compute_ap.

        This is a helper function of evaluate().
        """
        # Generate a temporary file.
        f = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = f.name
        f.writelines([all_names[i] + '\n' for i in ind])
        f.close()

        cmd = '%s %s %s' % (
            self._paths['compute_ap'],
            os.path.join(self._paths['groundtruth'], query_name), temp_filename)
        ap = os.popen(cmd).read()

        # Delete temporary file.
        os.remove(temp_filename)
        return float(ap.strip())


def main():
    """Main function of this program."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', dest='test', type=str, required=True,
                        help='Dataset to evaluate.')
    parser.add_argument('--via', dest='via', type=str, required=True,
                        help='Dataset to assistant PCA whitening.')
    parser.add_argument('--crop', dest='crop', type=bool, required=True,
                        help='Whether to use cropped query.')
    parser.add_argument('--dim', dest='dim', type=int, required=True,
                        help='Dimension of the final retrieval descriptor.')
    parser.add_argument('--whiten', dest='whiten', type=bool, required=True,
                        help='Whether to use whitening.')
    args = parser.parse_args()
    if args.test not in ['oxford', 'paris']:
        raise AttributeError('--test parameter must be oxford/paris.')
    if args.via not in ['oxford', 'paris']:
        raise AttributeError('--via parameter must be oxford/paris.')
    if args.dim <= 0:
        raise AttributeError('--dim parameter must >0.')

    test_image_root = os.path.join('/data/zhangh/data/', args.test)
    via_image_root = os.path.join('/data/zhangh/data/', args.via)
    project_root = '/data/zhangh/project/ddt/'
    test_data_root = os.path.join('/data/zhangh/project/ddt/data/', args.test)
    via_data_root = os.path.join('/data/zhangh/project/ddt/data/', args.via)
    paths = {
        'groundtruth': os.path.join(test_image_root, 'groundtruth/'),
        'test_all_pool5': os.path.join(test_image_root, 'pool5/all/'),
        'test_crop_pool5': os.path.join(test_image_root, 'pool5/crop/'),
        'test_full_pool5': os.path.join(test_image_root, 'pool5/full/'),
        'via_all_pool5': os.path.join(via_image_root, 'pool5/all/'),
        'test_all_pc1': os.path.join(test_data_root, 'pc1-refine/all/'),
        'test_crop_pc1': os.path.join(test_data_root, 'pc1-refine/crop/'),
        'test_full_pc1': os.path.join(test_data_root, 'pc1-refine/full/'),
        'via_all_pc1': os.path.join(via_data_root, 'pc1-refine/all/'),
        'compute_ap': os.path.join(project_root, 'lib/compute_ap'),
    }
    for k in paths:
        if k != 'compute_ap':
            assert os.path.isdir(paths[k])
        else:
            assert os.path.isfile(paths[k])

    evaluate_manager = EvaluateManager(paths)
    evaluate_manager.fitPca(dim=args.dim, whiten=args.whiten)
    evaluate_manager.evaluate(args.crop)


if __name__ == '__main__':
    main()
