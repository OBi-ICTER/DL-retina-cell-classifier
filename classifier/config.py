class Config(object):

 LOOP=10

 SEQ_SIZE=(256,256)

 TL_SIZE=(160,160)
 
 BATCH_SIZE=32

 dataset_train={'small':'datasets/cells/train', 
        'extended':'datasets/cells2/train',
        'amacrine':'datasets/cells1/train/amacrine',
        'bipolar':'datasets/cells1/train/bipolar',
        'cone':'datasets/cells1/train/cone',
        'ganglion':'datasets/cells1/train/ganglion',
        'horizontal':'datasets/cells1/train/horizontal',
        'muller':'datasets/cells1/train/muller',
        'rod':'datasets/cells1/train/rod',
        'rpe':'datasets/cells1/train/rpe'}

 dataset_evaluate={'small':'datasets/cells/validate', 
        'extended':'datasets/cells2/validate',
        'amacrine':'datasets/cells1/validate/amacrine',
        'bipolar':'datasets/cells1/validate/bipolar',
        'cone':'datasets/cells1/validate/cone',
        'ganglion':'datasets/cells1/validate/ganglion',
        'horizontal':'datasets/cells1/validate/horizontal',
        'muller':'datasets/cells1/validate/muller',
        'rod':'datasets/cells1/validate/rod',
        'rpe':'datasets/cells1/validate/rpe'}