# VASR: Visual Analogies of Situation Recognition

Repository for the paper "VASR: Visual Analogies of Situation Recognition": https://www.anonymous.co.il/  
To create the dataset, enter the [dataset](dataset) directory.  
To run model experiments, enter the [experiments](experiments) directory.

## Setup
- Run:
    ```
    virtualenv venv --python=python3.7  
    source venv/bin/activate
    pip install -r requirements.txt
    export PYTHONPATH=$(pwd)/dataset:$(pwd)/experiments
    ```
- Run installation script:
    ```shell
    ./install.sh 
    ```

## Abstract
A core process in human cognition is analogical mapping:
the ability to identify a similar relational structure between different sit-
uations. We introduce a new task, Visual Analogies of Situation Recog-
nition, adapting the word analogy task into the visual domain. Given
a triplet of images (A : A’ :: B :?), the task is to select an image can-
didate B’ that completes the analogy. Unlike previous work on visual
analogy, we focus on complex images describing scenes. We leverage sit-
uation recognition annotations and the CLIP model to generate over
500,000 silver-label analogies. We crowd-source annotations to create
a gold-standard test-set, achieving high human accuracy (90%). We ex-
periment with several baseline models and find that while some do well
when answer distractors are chosen randomly (∼86%), all struggle with
carefully selected distractors (∼53%). We hope our dataset will drive the
development of models with better analogy-making abilities
