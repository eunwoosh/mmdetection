# hyper parameters
num_classes = 20  # TODO need to align with dataset
# classes = ["object"]  # TODO need to align with dataset

# model.py
model = dict(
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
    backbone=dict(
        type='mobilenetv2_w1',
        out_indices=(2, 3, 4, 5),
        frozen_stages=-1,
        norm_eval=False,
        pretrained=True),
    type='CustomATSS',
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 320],
        out_channels=64,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CustomATSSHead',
        num_classes=num_classes,
        in_channels=64,
        stacked_convs=4,
        feat_channels=64,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        use_qfl=False,
        qfl_cfg=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0)))
load_from = 'https://storage.openvinotoolkit.org/repositories/openvino_training_extensions/models/object_detection/v2/mobilenet_v2-atss.pth'
resume_from = None
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
task = 'detection'

# added part
log_level = 'DEBUG'  # The level of logging.
workflow = [('train', 1)]  # Workflow for runner. [('train', 1)] means there is only one 
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=66) # Runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`
optimizer = dict(  # Config used to build optimizer, support all the optimizers in PyTorch whose arguments are also the same as those in PyTorch
    type='SGD',
    lr=0.004,
    momentum=0.9,  # Momentum
    weight_decay=0.0001)  # Weight decay of SGD
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=1e-06, by_epoch=True, warmup='linear', warmup_iters=3, warmup_ratio=0.333333)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook', by_epoch=True)])

# data_pipeline.py
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=[(992, 736), (896, 736), (1088, 736), (992, 672),
                   (992, 800)],
        multiscale_mode='value',
        keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(992, 736),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[0, 0, 0],
                std=[255, 255, 255],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        # type='VitensDataset',
        # ann_file='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/annotations/instances_train.json',
        # img_prefix='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/images/train/',
        type='VOCDataset',
        img_prefix='/home/eunwoo/work/data/voc/VOC2012/',
        ann_subdir='/home/eunwoo/work/data/voc/VOC2012/Annotations',
        ann_file='/home/eunwoo/work/data/voc/VOC2012/ImageSets/Main/train.txt',
        pipeline=train_pipeline),
    val=dict(
        # type='VitensDataset',
        # ann_file='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/annotations/instances_val.json',
        # img_prefix='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/images/val/',
        type='VOCDataset',
        img_prefix='/home/eunwoo/work/data/voc/VOC2012/',
        ann_subdir='/home/eunwoo/work/data/voc/VOC2012/Annotations',
        ann_file='/home/eunwoo/work/data/voc/VOC2012/ImageSets/Main/val.txt',
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        # type='VitensDataset',
        # ann_file='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/annotations/instances_val.json',
        # img_prefix='/home/eunwoo/work/exp_resource/dataset/vitens-tiled-coco/12/images/val/',
        type='VOCDataset',
        img_prefix='/home/eunwoo/work/data/voc/VOC2012/',
        ann_subdir='/home/eunwoo/work/data/voc/VOC2012/Annotations',
        ann_file='/home/eunwoo/work/data/voc/VOC2012/ImageSets/Main/val.txt',
        test_mode=True,
        pipeline=test_pipeline)
)
