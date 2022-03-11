import os

# ------------------------------Names--------------------------------

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'

SUPERVISED_CONCAT = 'supervised_concat'
SUPERVISED_ARITHMETIC = 'supervised_arithmetic'

model_description_options = {

    # (A,B,C) --> D
    SUPERVISED_CONCAT,
    # (C+(B-A)) --> D
    SUPERVISED_ARITHMETIC
}


# ------------------------------Constants--------------------------------

MODELS_MAP = {
    # https://arxiv.org/abs/2010.11929
    # ViT-Large model (ViT-L/32), ImageNet-1k weights fine-tuned from in21k @ 384x384
    'vit': 'vit_large_patch32_384',

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

IMSITU_PATH = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data'
IMAGES_PATH = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data/images_512'
SWIG_PREDICTIONS = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data/analogies_outputs/data/GSR_pred_parsed_results.json'

SOURCE = '/Users/eliyahustrugo/PycharmProjects/vasr/experiments'
SPLIT_PATH = os.path.join(SOURCE, 'date_split')
GOLD_PATH_DIR = os.path.join(SPLIT_PATH, 'gold_standard')
TEST_RANDOM_PATH = os.path.join(SPLIT_PATH, 'split_random', 'test_random.csv')
TEST_GOLD_PATH = os.path.join(SPLIT_PATH, 'gold_standard', 'test_gold.csv')

# ------------------------------Paths to Results--------------------------------

ZEROSHOT_RESULTS_PATH = os.path.join(IMSITU_PATH, 'analogies_outputs/model_results/zero_shot')
TRAIN_RESULTS_PATH = os.path.join(IMSITU_PATH, 'analogies_outputs/model_results/train')
