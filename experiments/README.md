# Usage

It's possible to run the following examples with several models:

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

### Zero-Shot
To evaluate arithmetic zero-shot on the random distractors run:
```commandline
python experiments/run_zero_shot.py --model vit --split random  
```
To evaluate arithmetic zero-shot on the difficult distractors run:
```commandline
python experiments/run_zero_shot.py --model vit --split difficult  
```

To use other model, just change the `--model`. For example to use Swin Transformer on the random distractors run:
```commandline
python experiments/run_zero_shot.py --model swin --split random  
```

### Trainable

#### To train the supervised concat model :
- on the random distractors run:

```commandline
python experiments/run_trainable.py --model_desc supervised_concat --model_backend_type vit --split random
```
- on the difficult distractors run:
```commandline
python experiments/run_trainable.py --model_desc supervised_concat --model_backend_type vit --split difficult
```

#### To train the supervised arithmetic model :
- on the random distractors run:

```commandline
python experiments/run_trainable.py --model_desc supervised_arithmetic --model_backend_type vit --split random
```
- on the difficult distractors run:
```commandline
python experiments/run_trainable.py --model_desc supervised_arithmetic --model_backend_type vit --split difficult
```