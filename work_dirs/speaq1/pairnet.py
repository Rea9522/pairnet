dataset_type = 'PanopticSceneGraphDataset'
ann_file = './data/psg/psg.json'
coco_root = './data/coco'
seg_root = './data/coco/annotations'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadPanopticSceneGraphAnnotations',
        with_bbox=True,
        with_rel=True,
        with_mask=True,
        with_seg=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AutoAugment',
        policies=[[{
            'type':
            'Resize',
            'img_scale': [(480, 800), (512, 800), (544, 800), (576, 800),
                          (608, 800), (640, 800), (672, 800), (704, 800),
                          (736, 800), (768, 800), (800, 800)],
            'multiscale_mode':
            'value',
            'keep_ratio':
            True
        }],
                  [{
                      'type': 'Resize',
                      'img_scale': [(400, 800), (500, 800), (600, 800)],
                      'multiscale_mode': 'value',
                      'keep_ratio': True
                  }, {
                      'type': 'RelRandomCrop',
                      'crop_type': 'absolute_range',
                      'crop_size': (384, 600),
                      'allow_negative_crop': False
                  }, {
                      'type':
                      'Resize',
                      'img_scale': [(480, 800), (512, 800), (544, 800),
                                    (576, 800), (608, 800), (640, 800),
                                    (672, 800), (704, 800), (736, 800),
                                    (768, 800), (800, 800)],
                      'multiscale_mode':
                      'value',
                      'override':
                      True,
                      'keep_ratio':
                      True
                  }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=1),
    dict(type='RelsFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSceneGraphAnnotations', with_bbox=True, with_rel=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
            dict(
                type='ToDataContainer',
                fields=({
                    'key': 'gt_bboxes'
                }, {
                    'key': 'gt_labels'
                })),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='PanopticSceneGraphDataset',
        ann_file='./data/psg/psg.json',
        img_prefix='./data/coco',
        seg_prefix='./data/coco/annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadPanopticSceneGraphAnnotations',
                with_bbox=True,
                with_rel=True,
                with_mask=True,
                with_seg=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='AutoAugment',
                policies=[[{
                    'type':
                    'Resize',
                    'img_scale': [
                        (480, 800), (512, 800), (544, 800), (576, 800),
                        (608, 800), (640, 800), (672, 800), (704, 800),
                        (736, 800), (768, 800), (800, 800)
                    ],
                    'multiscale_mode':
                    'value',
                    'keep_ratio':
                    True
                }],
                          [{
                              'type': 'Resize',
                              'img_scale': [(400, 800), (500, 800),
                                            (600, 800)],
                              'multiscale_mode': 'value',
                              'keep_ratio': True
                          }, {
                              'type': 'RelRandomCrop',
                              'crop_type': 'absolute_range',
                              'crop_size': (384, 600),
                              'allow_negative_crop': False
                          }, {
                              'type':
                              'Resize',
                              'img_scale': [(480, 800), (512, 800), (544, 800),
                                            (576, 800), (608, 800), (640, 800),
                                            (672, 800), (704, 800), (736, 800),
                                            (768, 800), (800, 800)],
                              'multiscale_mode':
                              'value',
                              'override':
                              True,
                              'keep_ratio':
                              True
                          }]]),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=1),
            dict(type='RelsFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_rels', 'gt_masks'])
        ],
        split='train',
        all_bboxes=True),
    val=dict(
        type='PanopticSceneGraphDataset',
        ann_file='./data/psg/psg.json',
        img_prefix='./data/coco',
        seg_prefix='./data/coco/annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadSceneGraphAnnotations',
                with_bbox=True,
                with_rel=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
                    dict(
                        type='ToDataContainer',
                        fields=({
                            'key': 'gt_bboxes'
                        }, {
                            'key': 'gt_labels'
                        })),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='test',
        all_bboxes=True),
    test=dict(
        type='PanopticSceneGraphDataset',
        ann_file='./data/psg/psg.json',
        img_prefix='./data/coco',
        seg_prefix='./data/coco/annotations',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='LoadSceneGraphAnnotations',
                with_bbox=True,
                with_rel=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=1),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='ToTensor', keys=['gt_bboxes', 'gt_labels']),
                    dict(
                        type='ToDataContainer',
                        fields=({
                            'key': 'gt_bboxes'
                        }, {
                            'key': 'gt_labels'
                        })),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        split='test',
        all_bboxes=True),
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=1)
checkpoint_config = dict(interval=1, max_keep_ckpts=15)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'pretrain/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth'
resume_from = None
workflow = [('train', 1), ('val', 1)]
num_object_classes = 133
num_relation_classes = 56
find_unused_parameters = True
model = dict(
    type='PSGTr',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    bbox_head=dict(
        type='CrossHead3',
        num_classes=133,
        num_relations=56,
        num_obj_query=100,
        num_rel_query=100,
        mapper='conv_tiny',
        in_channels=[256, 512, 1024, 2048],
        feat_channels=256,
        out_channels=256,
        num_transformer_feat_level=3,
        embed_dims=256,
        enforce_decoder_input_project=False,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=False,
            num_layers=9,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))),
        relation_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=6,
            transformerlayers=dict(
                type='BaseTransformerLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.1,
                    dropout_layer=None,
                    add_identity=True),
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm'))),
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        rel_cls_loss=dict(
            type='SeesawLoss',
            num_classes=56,
            return_dict=True,
            loss_weight=2.0),
        subobj_cls_loss=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=4.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0
            ]),
        importance_match_loss=dict(
            type='BCEWithLogitsLoss', reduction='mean', loss_weight=5.0),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                1.0, 0.1
            ]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0)),
    train_cfg=dict(
        id_assigner=dict(
            type='SpeaQMatcher',
            sub_id_cost=dict(type='ClassificationCost', weight=1.0),
            obj_id_cost=dict(type='ClassificationCost', weight=1.0),
            r_cls_cost=dict(type='ClassificationCost', weight=0.0)),
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        mask_assigner=dict(
            type='MaskHungarianAssigner',
            cls_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')),
    test_cfg=dict(max_per_img=100))
custom_imports = dict(
    imports=[
        'pairnet.models.frameworks.psgtr', 'pairnet.models.losses.seg_losses',
        'pairnet.datasets', 'pairnet.datasets.pipelines.loading',
        'pairnet.datasets.pipelines.rel_randomcrop',
        'pairnet.models.relation_heads.approaches.matcher',
        'pairnet.models.relation_heads.pairnet_speaq_head', 'pairnet.utils'
    ],
    allow_failed_imports=False)
evaluation = dict(
    interval=100000000,
    metric='sgdet',
    relation_mode=True,
    classwise=True,
    iou_thrs=0.5,
    detection_method='pan_seg')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1, decay_mult=1),
            transformer_decoder=dict(lr_mult=0.1, decay_mult=1),
            pixel_decoder=dict(lr_mult=0.1, decay_mult=1),
            decoder_input_projs=dict(lr_mult=0.1, decay_mult=1)),
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy='step', gamma=0.5, step=[5, 10])
runner = dict(type='EpochBasedRunner', max_epochs=15)
project_name = 'ATM'
expt_name = 'speaq1'
work_dir = './work_dirs/speaq1'
auto_scale_lr = dict(enable=True, base_batch_size=1)
auto_resume = False
gpu_ids = [0]
