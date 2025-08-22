from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp


config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_s_qs"
config.resume = False
config.output = "output/VIT_S_FIQA_FR_ms1mv2_"
config.embedding_size = 384
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 256
config.optimizer = "adamw"
config.lr = 1e-3 #3e-4
config.verbose = 2000
config.dali = False
config.save_all_states=True
config.frequent=50
config.alpha=100
config.dropout=0.0
config.augmentation = "gridsample"
config.gridaug_params = {'scale_min': 0.8,
              'scale_max': 1.2,
              'rot_prob': 0.2,
              'max_rot': 20,
              'hflip_prob': 0.5,
              'extra_offset': 0.1,
              'photometric_num_ops': 2,
              'photometric_magnitude': 14,
              'photometric_magnitude_offset': 9,
              'photometric_num_magnitude_bins': 31,
              'blur_magnitude': 1.0,
              'blur_prob': 0.2,
              'cutout_prob': 0.2
                  }


config.rec = "/data/Biometrics/database/faces_webface_112x112"#"data/ms1mv2_112x112.lmdb_dataset"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 20
config.warmup_epoch = config.num_epoch // 8
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
