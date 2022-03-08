#Usage
##Config
First set the following paths in the experiments/config.py:
```python
# Path to the Directory containing images
IMAGES_PATH = 'path/to/dir'
#Path to the Directory containing the split files
SPLIT_PATH = 'path/to/dir'
```
##Zero-Shot
To evaluate arithmetic zero-shot on the random distractors run:
```
python run_zero_shot.py --model vit --split random  
```
To evaluate arithmetic zero-shot on the difficult distractors run:
```
python run_zero_shot.py --model vit --split difficult  
```
It's possible to run  the above examples with the following models:

```python
{
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

```
For exmaple, to use Swin Transmofer on the random distractors run:
```
python run_zero_shot.py --model swin --split random  

```

##Trainable


