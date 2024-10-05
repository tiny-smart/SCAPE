log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=10, metric='PCK', key_indicator='PCK', gpu_collect=True)
optimizer = dict(
    type='Adam',
    lr=2e-4,
     paramwise_cfg=dict(
        custom_keys={'encoder_sample': dict(lr_mult=0.1)})
)




optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.001,
    min_lr_ratio=1e-5)
total_epochs = 190



log_config = dict(
    interval=50,
    hooks=[
      dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])



channel_cfg = dict(
    num_output_channels=1,
    dataset_joints=1,
    dataset_channel=[
        [0, ],
    ],
    inference_channel=[
        0,
    ],
    max_kpt_num=100)



extra= dict(
            PRETRAINED_LAYERS=('conv1', 'bn1','conv2','bn2','layer1'
                                   ,'transition1','stage2','transition2','stage3'),
            FINAL_CONV_KERNEL=1,
            STAGE2=dict(
                NUM_MODULES=1,
                NUM_BRANCHES=2,
                BLOCK='BASIC',
                NUM_BLOCKS=(4, 4),
                NUM_CHANNELS=(32, 64),
            FUSE_METHOD='SUM'),
            STAGE3=dict(
            NUM_MODULES=4,
            NUM_BRANCHES=3,
            BLOCK='BASIC',
            NUM_BLOCKS=(4, 4, 4),
            NUM_CHANNELS=(32, 64, 128),
           FUSE_METHOD= 'SUM'))









# model settings

model = dict(
    type='TransformerPose',
    pretrained='torchvision://resnet50',
    encoder_sample=dict(type='ResNet', depth=50),
    encoder_query=dict(type='ResNet', depth=50),
    keypoint_head=dict(
        type='TokenPose_TB_base',
        in_channels=2048,
        transformer=dict(
            type='Transformer',
            dim=256,depth=4,heads=8,
            mlp_dim=256*2, dropout=0.1,num_keypoints=100,
            all_attn=False, scale_with_head=True),
            train_cfg=None,
            test_cfg=None,
            dim=256,
            hidden_heatmap_dim=64 * 6,
            heatmap_dim=64 * 64,
            apply_multi=True,
            apply_init=True,
            loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True)),
    train_cfg=dict(),
        test_cfg=dict(
            flip_test=False,
            post_process='default',
            shift_heatmap=True,
            modulate_kernel=11))



data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel']
)



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=15, scale_factor=0.15),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs', 'category_id','flip','pair'
        ]),

]



valid_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffineFewShot'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTargetFewShot', sigma=2),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file','joints_3d', 'joints_3d_visible', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs', 'category_id','flip','pair'
        ]),
]



test_pipeline = valid_pipeline

data_root = 'data/mp100'
data = dict(
   samples_per_gpu=16,
    workers_per_gpu=8,
    train=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_train.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        pipeline=train_pipeline),
    val=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_val.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=100,
        pipeline=valid_pipeline),

    test=dict(
        type='TransformerPoseDataset',
        ann_file=f'{data_root}/annotations/mp100_split1_test.json',
        img_prefix=f'{data_root}/',
        data_cfg=data_cfg,
        valid_class_ids=None,
        max_kpt_num=channel_cfg['max_kpt_num'],
        num_shots=1,
        num_queries=15,
        num_episodes=200,
        pipeline=valid_pipeline),
)



shuffle_cfg = dict(interval=1)