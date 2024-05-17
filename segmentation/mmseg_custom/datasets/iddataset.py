# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset

@DATASETS.register_module(force=True)
class IDDataset(CustomDataset):
    """Buidlings dataset.

    In segmentation map annotation for Potsdam dataset, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """
    CLASSES = ('road', 'parking', 'drivable fallback',
           'sidewalk', 'rail track', 'non-drivable fallback',
           'person', 'animal', 'rider', 'motorcycle', 'bicycle',
           'autorickshaw', 'car', 'truck', 'bus', 'caravan', 'trailer',
           'train', 'vehicle fallback', 'curb', 'wall',
           'fence', 'guard rail', 'billboard', 'traffic sign',
           'unlabeled')

    PALETTE = [
    [128, 64, 128], [72, 98, 91], [255, 204, 54], [220, 20, 60], [147, 114, 178],
    [132, 91, 83], [70, 70, 70], [105, 143, 35], [255, 69, 0], [0, 191, 255],
    [128, 0, 128], [255, 165, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [0, 255, 255], [255, 0, 255], [192, 192, 192], [255, 192, 203],
    [75, 0, 130], [255, 99, 71], [100, 149, 237], [255, 140, 0], [0, 255, 127],
    [255, 20, 147]
]

    def __init__(self, **kwargs):
        super(IDDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
