import os

# ------------------------------Names--------------------------------

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'
# ------------------------------Constants--------------------------------
FEW_SHOT_DATA_SAMPLES = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

MODELS_MAP = {
    # https://arxiv.org/abs/2010.11929
    # ViT-Large model (ViT-L/32), ImageNet-1k weights fine-tuned from in21k @ 384x384
    'vit': 'vit_small_patch32_384',

    # https://arxiv.org/abs/2103.14030
    # Swin-L @ 384x384, pretrzained ImageNet-22k, fine tune 1k
    'swin': 'swin_large_patch4_window12_384',

    # https://arxiv.org/abs/2012.12877
    # DeiT base model @ 384x384,ImageNet-1k weights from https://github.com/facebookresearch/deit.
    'deit': 'deit_base_patch16_384',

    # https://arxiv.org/pdf/2201.03545.pdf
    'convnext': 'convnext_large'

}

# ------------------------------Paths--------------------------------

IMAGES_PATH = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data/images_512'
SPLIT_PATH = '/Users/eliyahustrugo/PycharmProjects/vasr/experiments/date_split'

GOLD_PATH_DIR = os.path.join(SPLIT_PATH, 'gold_standard')
TEST_RANDOM_PATH = os.path.join(SPLIT_PATH, 'split_random', 'test_random.csv')
TEST_GOLD_PATH = os.path.join(SPLIT_PATH, 'gold_standard', 'test_gold.csv')

# ------------------------------Paths to Results--------------------------------

ZEROSHOT_RESULTS_PATH = 'experiments/model_results/zero_shot'
TRAIN_RESULTS_PATH = 'experiments/model_results/train'
