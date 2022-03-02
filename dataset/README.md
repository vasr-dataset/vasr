# VASR: Visual Analogies Of Situation Recognition

To build the VASR dataset we leverage situation recognition annotations from imSitu. 
We start by finding likely image candidates based on the gold annotated frames and finding challenging distractors. 
We apply several filters in order to keep pairs of images with a single salient difference between them.
We then choose candidates for the gold test set and annotate a gold dataset of analogies that humans are able to solve correctly.
Finally, we provide the final dataset statistics.

See [our paper](https://www.google.co.il/) for details, and the [dataset webpage](https://vasr-dataset.github.io/) for exploring our dataset.

## Download dataset

The dataset is available on the website:
* [Download dataset](https://vasr-dataset.github.io/download) 
<br/>
  <!--The following splits are available for download: Entire dataset, Silver train, Silver test, Gold train, Gold dev and Gold test.--> 


## Generate dataset

In order to generate the dataset, one should run the pipeline, i.e. run all the files in the pipeline folder.




