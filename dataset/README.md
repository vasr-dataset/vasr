# VASR: Visual Analogies Of Situation Recognition

To build the VASR dataset we leverage situation recognition annotations from imSitu. 
We start by finding likely image candidates based on the gold annotated frames and finding challenging distractors. 
We apply several filters in order to keep pairs of images with a single salient difference between them.
We then choose candidates for the gold test set and annotate a gold dataset of analogies that humans are able to solve correctly.

See [our paper](https://www.google.co.il/) for details, and the [dataset webpage](https://www.google.co.il/) for exploring our dataset.

# Download dataset

### The dataset is available on the website:
* [Download dataset](https://www.google.co.il/) 
<br/>
  <!--The following splits are available for download: Entire dataset, Silver train, Silver test, Gold train, Gold dev and Gold test.--> 


# Generate dataset

### To generate the dataset, run the entire pipeline: all the files in the pipeline folder.

## Requirements 
#### 1. Download [Imsitu project](https://github.com/my89/imSitu) and change the path in config file accordingly.
#### 2. Download [SWIG project](https://github.com/swig/swig) and change the path in config file accordingly.
#### 3. Change the `SPLIT` variable in the config.py to the desire split (train, testdev).

**Convention**: In the paper we use A:A' :: B:B', in the code we use A:B :: C:D. 

## Pipeline 
#### 1. `A_find_AB_pairs.py`
Finding images A:A' that are annotated the same, except of a single different role (Section 3.1 in the paper: Finding Analogous Situations in imSitu).  

Run example:
`python dataset/pipeline/A_find_AB_pairs.py`

#### 2. `B_textual_filter.py`
Textual filter that leverages WordNet and FrameNet to filter ambiguous image pairs (Section 3.3, Over-specified annotations). 

#### 3. `C_filter_visual.py`
Visual filter that leverages SWiG to filter to filter images with non visually salient object (Section 3.3, Over-specified annotations). 

#### 4. `D_extract_clip_features.py`
CLIP based vision-and-language filter that filters ambiguous image pairs (Section 3.3, Under-specified annotations). 
We aim to filter cases of such ambiguity, where an object can describe the _other_ image bounding box.

#### 5. `E_classify_ambiguous_image_pairs.py`
Same description as before. `D_extract_clip_features.py` is used for extraction, and `E_classify_ambiguous_image_pairs.py` for the filtering given the extracted features. 

#### 6. `F_process_and_save_ab_pairs.py`
Save in cache all of the filtered A:B pairs. In next step we iterate on it to find C:D pairs. 

#### 7. `H_create_splits.py` 
Creating the final data, composing train, dev, and test splits (Sections 3.4 Building the Test Set + 3.6 Final Datasets and Statistics). 

#### 8. `I_extract_candidates_for_clip_filter.py`
Extracting candidates to difficult distractors (Section 3.2 Choosing Difficult Distractors). 

#### 9. `J_extract_clip_features_for_distractors.py`
Extracting CLIP features to the received distractors candidates.

#### 10. `K_classify_and_pack_distractors.py`
Choosing the final distractors by filtering the ambiguous distractors (Section 3.3, Under-specified annotations).

#### 11. `L_pack_splits_with_random_images.py`
Chossing random distractors.
