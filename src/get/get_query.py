#!/usr/bin/env python
"""Exatract cropped query images for Oxford5k and Paris6k datasets.

This program is modified from CroW:
    https://github.com/yahoo/crow/
"""


import glob
import os

import PIL.Image


__all__ = ['CropManager']
__author__ = 'Hao Zhang'
__copyright__ = '2017 LAMDA'
__date__ = '2017-10-27'
__email__ = 'zhangh0214@gmail.com'
__license__ = 'CC BY-SA 3.0'
__status__ = 'Development'
__updated__ = '2017-11-24'
__version__ = '2.0'


class CropManager(object):
    """Exatract cropped query images.

    Attributes:
        _dataset, str: Name of the dataset, either 'oxford5k' or 'paris6k'.
        _paths, dict of (str, str): Image data and feature paths.
    """
    def __init__(self, paths, dataset):
        print('Initiate.')
        self._paths = paths
        self._dataset = dataset

    def getCrop(self):
        """Extract cropped query images from Oxford5k or Paris6k dataset.

        The cropped query images are save in path self._paths['crop_image'].
        """
        print('Exact cropped queries from database images.')
        for f in glob.iglob(os.path.join(self._paths['groundtruth'],
                                         '*_query.txt')):
            query_name = os.path.splitext(
                os.path.basename(f))[0].replace('_query', '')
            image_name, x, y, w, h = open(f).read().strip().split(' ')
            if self._dataset == 'oxford5k':
                image_name = image_name.replace('oxc1_', '')

            # Fetch cropped query region from the image.
            image = PIL.Image.open('%s.jpg' % os.path.join(
                self._paths['all_image'], image_name))
            x, y, w, h = map(float, (x, y, w, h))
            box = map(lambda d: int(round(d)), (x, y, x + w, y + h))
            image = image.crop(box)
            image.save('%s.png' % os.path.join(self._paths['crop_image'],
                                               query_name))


def main():
    """Main function of the program,"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset', type=str, required=True,
                        help='Dataset to evaluate.')
    args = parser.parse_args()
    if args.dataset not in ['oxford5k', 'paris6k']:
        raise AttributeError('--dataset paramete must be oxford5k/paris6k.')

    project_root = os.popen('pwd').read().strip()
    data_root = os.path.join(os.path.join(project_root, 'data'), args.dataset)
    paths = {
        'all_image': os.path.join(data_root, 'image/all/'),
        'crop_image': os.path.join(data_root, 'image/crop/'),
        'groundtruth': os.path.join(data_root, 'groundtruth/'),
    }
    for k in paths:
        assert os.path.isdir(paths[k])

    manager = CropManager(paths, args.dataset)
    manager.getCrop()


if __name__ == '__main__':
    main()
