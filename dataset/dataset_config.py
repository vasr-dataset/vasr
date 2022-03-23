SPLIT = 'test'  # train, testdev, dev, test
PARAMS_FOR_SPLIT = {'train': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'dev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'testdev': {'MAX_CDS_MATCHES_FOR_AB': 10, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 10, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 40, 'MAX_CLIP_CD_FILTER': 100},
                    'test': {'MAX_CDS_MATCHES_FOR_AB': 1, 'MAX_CDS_MATCHES_FOR_AB_SAMPLE_FROM': 1, 'MAX_OCC_FOR_EACH_IMAGE_IN_AB_PAIR': 4, 'MAX_CLIP_CD_FILTER': 100},}
BBOX_PCT_THRESHOLD = 0.02

print(f"*** SPLIT: {SPLIT} ***")

imsitu_path = 'dataset/assets/imsitu_splits'
swig_path = 'dataset/assets/swig_splits'
analogies_output_path = 'dataset/analogies_output'
swig_images_path = 'images_512'
