_base_ = "/media/ram/338f6363-03b7-4ad7-a2be-40c31f59dee4/20230418_backup/ram/Students/B.Tech_2020/sasank/InternImage/segmentation/configs/cityscapes/upernet_internimage_t_512x1024_160k_cityscapes.py"
num_classes=26
dataset_type = "IDDataset"
data_root = 'data/idd/'
classes = ('road', 'parking', 'drivable fallback',
           'sidewalk', 'rail track', 'non-drivable fallback',
           'person', 'animal', 'rider', 'motorcycle', 'bicycle',
           'autorickshaw', 'car', 'truck', 'bus', 'caravan', 'trailer',
           'train', 'vehicle fallback', 'curb', 'wall',
           'fence', 'guard rail', 'billboard', 'traffic sign',
           'unlabeled')

data=dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type = dataset_type,
        data_root = data_root,
        img_dir='leftImg8bit/train/',
        ann_dir='gtFine/train/',
        ),
    val=dict(
        type = dataset_type,
        data_root = data_root,
        img_dir='leftImg8bit/val/',
        ann_dir='gtFine/val/'),
    test=dict(
        type = dataset_type,
        data_root = data_root,
        img_dir='leftImg8bit/val/',
        ann_dir='gtFine/val/'))


