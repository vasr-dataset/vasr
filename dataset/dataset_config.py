import io
from copy import deepcopy

import PIL
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import random
import os
import numpy as np
from PIL import Image

SPLIT = 'train'  # train, testdev
PARAMS_FOR_SPLIT = {'train': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'dev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'testdev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'test': {'MAX_CDS_MATCHES_FOR_AB': 1, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 1, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 4, 'MAX_CLIP_CD_FILTER': 100},}
NUM_CANDIDATES = 4
FONT_SIZE = 16
BBOX_PCT_THRESHOLD = 0.02

print(f"*** #### SPLIT: {SPLIT}, PARAMS_FOR_SPLIT: {PARAMS_FOR_SPLIT[SPLIT]} ### ***")

# imsitu_path = r'C:\devel\image_analogies\imSitu'
# swig_path = r'C:\devel\vasr\dataset\assets\splits'
# swig_images_path = r'C:\devel\swig\images\images_512'
imsitu_path = '/data/users/yonatab/analogies/vasr/dataset/assets/imsitu_splits'
swig_path = '/data/users/yonatab/analogies/vasr/dataset/assets/swig_splits'
analogies_output_path = '/data/users/yonatab/analogies/vasr/dataset/analogies_output'
swig_images_path = '/data/users/yonatab/analogies/swig/images_512'
