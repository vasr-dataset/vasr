# VASR: Visual Analogies Of Situation Recognition

To build the VASR dataset we leverage situation recognition annotations from imSitu. 
We start by finding likely image candidates based on the gold annotated frames and finding challenging distractors. 
We apply several filters in order to keep pairs of images with a single salient difference between them.
We then choose candidates for the gold test set and annotate a gold dataset of analogies that humans are able to solve correctly.
Finally, we provide the final dataset statistics.

See [our paper](https://www.google.co.il/) for details, and the [dataset webpage](https://vasr-dataset.github.io/) for exploring our dataset.

# Download dataset

### The dataset is available on the website:
* [Download dataset](https://vasr-dataset.github.io/download) 
<br/>
  <!--The following splits are available for download: Entire dataset, Silver train, Silver test, Gold train, Gold dev and Gold test.--> 


# Generate dataset

### To generate the dataset, one should run the entire pipeline, i.e. run all the files in the pipeline folder.

## Requirements 
#### 1. Download [Imsitu project](https://github.com/my89/imSitu) and change the path in config file accordingly.
#### 2. Download [SWIG project](https://github.com/swig/swig) and change the path in config file accordingly.
#### 3. Change the `SPLIT` variable in the config.py to the desire split (dev, train, test).


## Pipeline 
#### 1. A_find_AB_pairs.py 
#### 1. B_filter_textual.py
#### 1. C_filter_visual.py <!-- after run python merge_clip_VL_feats_for_AB_filter.py # if ran CLIP extract with multiple processes, merge the feats with this-->
#### 1. E_process_and_save_AB_pairs.py
#### 1. F_find_CD_pairs.py -V
#### 1. G_create_train_full_and_ood.py - G_create-splits ?
#### 1. H ?
#### 1. I_extract_clip_features_for_distractors.py <!-- For each SPLIT in ['train', 'dev', 'test] -->
#### 1. J_classify_and_pack_distractors.py <!-- before run python merge_clip_VL_feats_for_distractors_filter.py # if ran CLIP extract with multiple processes, merge the feats with this - X -->
#### 1. K_pack_splits_with_random_images.py
#### 1. L ?
#### 1. M ?




