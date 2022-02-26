import os

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'
IMSITU_PATH = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data'
IMAGES_PATH = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data/images_512'
SWIG_PREDICTIONS = '/Users/eliyahustrugo/PycharmProjects/image_analogies/imsitu_data/analogies_outputs/data/GSR_pred_parsed_results.json'

SOURCE = os.path.abspath(os.getcwd())
SPLIT_PATH = os.path.join(SOURCE, 'experiments/date_split')

TEST_RANDOM_PATH = os.path.join(SPLIT_PATH, 'split_random', 'test_random.csv')
TEST_GOLD_PATH = os.path.join(SPLIT_PATH, 'gold_standard', 'test_gold.csv')

MODEL_RESULTS_PATH = os.path.join(IMSITU_PATH, 'analogies_outputs/model_results')
ZEROSHOT_RESULTS_PATH = os.path.join(IMSITU_PATH, 'analogies_outputs/model_results/zero_shot')
TRAIN_RESULTS_PATH = os.path.join(IMSITU_PATH, 'analogies_outputs/model_results/train')

for d in [MODEL_RESULTS_PATH, ZEROSHOT_RESULTS_PATH, TRAIN_RESULTS_PATH]:
    if not os.path.exists(d):
        os.makedirs(d)

TEST_DISTRACTORS_PATH = os.path.join(SPLIT_PATH, 'split_distractors', 'analogies_distractors_test_final.csv')
