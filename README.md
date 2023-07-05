# PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation
 - [News](#News)
 - [Introduction](#Introduction)
 - [Getting Started](#Getting-Started)
 - [FSD50K Recipe](#FSD50K-Recipe)
 - [AudioSet Recipe](#Audioset-Recipe)
 - [Label Enhancement](#Label-Enhancement)
 - [Ensemble and Weight Averaging](#Ensemble-and-Weight-Averaging)
 - [Pretrained Models](#Pretrained-Models)
 - [Pretrained Enhanced Label Sets](#Pretrained-Enhanced-Label-Sets)
 - [Use Pretrained Model for Audio Tagging Inference in One-Click](#Use-Pretrained-Model-for-Audio-Tagging-Inference-in-One-Click)
 - [Use PSLA Training Pipeline For New Models](#Use-PSLA-Training-Pipeline-For-New-Models)
 - [Use PSLA Training Pipeline For New Datasets and Tasks](#Use-PSLA-Training-Pipeline-For-New-Datasets-and-Tasks)
 - [Use Pretrained CNN+Attention Model For New Tasks](#Use-Pretrained-CNN+Attention-Model-For-New-Tasks)
 - [Contact](#Contact)

## News
* April 2022: I will present PSLA at [13 May (Friday), 10:00 - 10:45 AM, New York Time at ICASSP 2022](https://2022.ieeeicassp.org/view_paper.php?PaperNum=9274).

## Introduction

<p align="center"><img src="https://raw.githubusercontent.com/YuanGongND/psla/main/fig/psla_poster_rs.png" alt="Illustration of PSLA." width="1050"/></p>

This repository contains the official implementation (in PyTorch) of the **PSLA Training Pipeline and CNN+Attention Model** proposed in the TASLP paper [PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation](https://arxiv.org/pdf/2102.01243.pdf) (Yuan Gong, Yu-An Chung, and James Glass, MIT).  

**PSLA** is a strong training pipeline that can significantly improve the performance of all models we evaluated (by 24.2% on average) on AudioSet and FSD50K. By applying PSLA on a **CNN+Attention** model, we achieved new state-of-the-art results on both AudioSet and FSD50 while the model only has approximately 16% parameters compared with the previous state-of-the-art model ([PANNs](https://arxiv.org/abs/1912.10211)) in early 2021. The model is still the best CNN based model now.

This repo can be used for multiple purposes:
* If you are not interested in audio tagging research, but just want to use the pretrained model for audio tagging applications, we provide a script to do it in almost one-click. We support unlimited length audio (e.g., hour-level). Please see [here](#Use-Pretrained-Model-for-Audio-Tagging-Inference-in-One-Click).
* If you want to reproduce the results in the PSLA paper, we provide the [AudioSet Recipe](#AudioSet-Recipe) and [FSD50K Recipe](#FSD50K-Recipe) for easy reproduction. We also provide our training logs and all [pretrained models](#Pretrained-Models).
* If you want to take a closer look at the PSLA training pipeline, all codes are in the ``src`` directory. We provide instruction for [label_enhancement](#Label-Enhancement) and [ensemble](#Ensemble-and-Weight-Averaging).
* If you want to use the PSLA training pipeline for your own model or your own task, please see [here](#Use-PSLA-Training-Pipeline-For-New-Models) and [here](#Use-PSLA-Training-Pipeline-For-New-Datasets-and-Tasks).
* If you want to use the pretrained model for new tasks, we provide the AudioSet and FSD50K pretrained models, please see [here](#Use-Pretrained-CNN+Attention-Model-For-New-Tasks).
* If you want to use the enhanced label set of AudioSet (both balanced and full training set) and FSD50K, we provide [pretrained enhanced label set](#Pretrained-Enhanced-Label-Sets) that can be used as drop-in replacement of original label set. With the enhanced label set, the performance of models trained with balanced AudioSet training set and FSD50K dataset can be improved.

Please cite our paper if you find this repository useful.

```  
@ARTICLE{gong_psla, 
    author={Gong, Yuan and Chung, Yu-An and Glass, James},  
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
    title={PSLA: Improving Audio Tagging with Pretraining, Sampling, Labeling, and Aggregation},   
    year={2021}, 
    doi={10.1109/TASLP.2021.3120633}
}
```  
  
## Getting Started  

Clone or download this repository and set it as the working directory, create a virtual environment and install the dependencies. We use Python 3.7.4.

```
cd psla/ 
python3 -m venv venv-psla
source venv-psla/bin/activate
pip install -r requirements.txt 
```

**Where's the code?**

The EfficientNet model file is in `src/models/Models.py`, the audio dataloader code is in `src/dataloaders/`, the training and evaluation code is in `src/traintest.py`, the main running file is in `src/run.py`. These codes are data-agnoistic. The label enhancement code is in `src/label_enhancement`, the ensemble code is in `src/ensemble`.

The recipes are in `egs/[audioset,fsd50k]`, when you run `run.sh`, it will call `src/run.py`, which will then call `src/dataloaders/audioset_dataset.py` and `/src/traintest.py`, which will then call `/src/models/Models.py`. For FSD50K, we provide the data pre-processing code in `egs/fsd50k/prep_fsd.py`, for AudioSet, you need to prepare it by yourself as you need to download audios from YouTube. `run.sh` contains all hyper-parameters of our experiment.

## FSD50K Recipe  
The FSD50K recipe is in `psla/egs/fsd50k/`. Note we use 16kHz sampling rate, which is lower than the original FSD50K sampling rate to lower the computational overhead. Please make sure you have installed the dependencies in `psla/requirement.txt`.

**Step 1. Download the FSD50K dataset from [the official website](https://zenodo.org/record/4060432).**

**Step 2. Prepare the data.**

Change the `fsd50k_path` in `egs/fsd50k/prep_fsd.py` (line 15) to your dataset path. And run:

```
cd psla/egs/fsd50k 
python3 prep_fsd.py
```

This should create three json files `fsd50k_tr_full.json`, `fsd50k_val_full.json`, and `fsd50k_eval_full.json` in `/egs/fsd50k/datafiles`. These will be used in training and evaluation. 

(Optional) **Step 3. Enhance the label of the FSD50K training set.**

Download our model prediction (or you can use yours after 1st round training) from [here](https://www.dropbox.com/s/kd84fq9ygwmidvp/prediction_files.zip?dl=1). Place it in `psla/src/label_enhancement/` and uncompress it.
```
cd psla/src/label_enhancement
python3 fix_type1.py
python3 fix_type2.py
python3 merge_type_1_2.py
```
This will automatically generate a set of new Json datafiles in `egs/fsd50k/datafiles`, e.g., `fsd50k_tr_full_type1_2_mean.json` means the datafile with enhanced label set for both Type-I and Type-II error with label modification threshold of mean. You can use these new datafiles as input of `egs/fsd50k/run.sh`, specifically, you can change `p` (label modification threshold) in `[none, mean, median, 5, 10, 25]`. 

If you skipped this step, please set `p` in `egs/fsd50k/run.sh` as `none`. You will get a slightly worse result.

**Step 4. Run the training and evaluation.**

``` 
cd psla/egs/fsd50k
(slurm user) sbatch run.sh
(local user) ./run.sh
```  

The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. The entire training and evaluation takes about 15 hours. Trimming the `target_length` of `run.sh` from 3000 to 1000, and increasing the batch size and learning rate accordingly can significantly reduce the running time, but leads to just slightly worse result.

**Step 5. Get the results.**

The running log will present the results of 1) single model, 2) weight averaging model (i.e., averaging the weight of last few model checkpoints, this does not increase the model size and computational overhead), and 3) checkpoint ensemble models (i.e., averaging the prediction of each epoch) on both the FSD50K official validation set and evaluation set. These results are also saved in `psla/egs/fsd50k/exp/yourexpname/[best_single_,wa_,ensemble_]result.csv`.

We share our training and evaluation log in `psla/egs/fsd50k/exp/`, you can expect to get a similar result as follows. We also share our entire experiment directory at this [dropbox link](https://www.dropbox.com/s/qfaeyuvtse420dn/demo-efficientnet-2-5e-4-fsd50k-impretrain-True-fm48-tm192-mix0.5-bal-True-b24-lemean-2.zip?dl=1). 

```
---------------evaluate best single model on the validation set---------------
mAP: 0.588115
AUC: 0.960351
---------------evaluate best single model on the evaluation set---------------
mAP: 0.558463
AUC: 0.943927
---------------evaluate weight average model on the validation set---------------
mAP: 0.587779
AUC: 0.960226
---------------evaluate weight averages model on the evaluation set---------------
mAP: 0.561647
AUC: 0.943910
---------------evaluate ensemble model on the validation set---------------
mAP: 0.601013
AUC: 0.970726
---------------evaluate ensemble model on the evaluation set---------------
mAP: 0.572588
AUC: 0.955053
```

## Audioset Recipe  

Audioset recipe is very similar with FSD50K recipe, but is a little bit more complex, you will need to prepare your data by yourself. The AudioSet recipe is in `psla/egs/audioset/`.

**Step 1. Prepare the data.**

Please prepare the json files (i.e., `train_data.json` and `eval_data.json`) by your self.
The reason is that the raw wavefiles of Audioset is not released and you need to download them by yourself. We have put a sample json file in `psla/egs/audioset/datafiles`, please generate files in the same format (You can also refer to `psla/egs/fsd50k/prep_fsd.py`). Please keep the label code consistent with `psla/egs/audioset/class_labels_indices.csv`.

Note: we use `16kHz` sampling rate for all AudioSet experiments.

Once you have the json files, you will need to generate the sampling weight file for full AudioSet json file.
```
cd psla/egs/audioset
python ../../src/gen_weight_file.py --dataset audioset --label_indices_path ./class_labels_indices.csv --datafile_path ./datafiles/yourdatafile.json
```

(Optional) **Step 2. Enhance the label of the balanced AudioSet training set.**

If you are experimenting with Full AudioSet, you can skip this, enhanced label does not improve the performance (potentially due to the evaluation label set is noisy). 

If you are experimenting with Balanced AudioSet or want to enhance the labelset anyway, check our [pretrained enhanced label set](https://github.com/YuanGongND/psla#pretrained-enhanced-label-set), and change the labels in your datafile.

**Step 3. Run the training and evaluation.**

Change the `data-train` and `data-val` in `psla/egs/audioset/run.sh` to your datafile path. 
Also change `subset` to [`balanced`,`full`] for balanced and full AudioSet, respectively.

``` 
cd psla/egs/audioset
(slurm user) sbatch run.sh
(local user) ./run.sh
```  

The recipe was tested on 4 GTX TITAN GPUs with 12GB memory. The entire training and evaluation takes about 12 hours for balanced AudioSet and about one week for full AudioSet. 

**Step 4. Get the results.**

The running log will present the results of 1) single model, 2) weight averaging model (i.e., averaging the weight of last few model checkpoints, this does not increase the model size and computational overhead), and 3) checkpoint ensemble models (i.e., averaging the prediction of each epoch) on AudioSet evaluation set. These results are also saved in `psla/egs/audioset/exp/yourexpname/[best_single_,wa_,ensemble_]result.csv`.

We share our training and evaluation log in `psla/egs/audioset/exp/`, you can expect to get a similar result as follows. We also share our entire experiment directory at this [dropbox link](). 

**Step 5. Reproducing the Ensemble Results in the PSLA paper.**

In step 4, only a single model is trained and only single run checkpoint ensemble is used. To reproduce the best ensemble results (0.474 mAP) in the PSLA paper, you need to run step 4 multiple times with same or different settings. To ease this process, we provide pretrained model of all ensemble models. You can download them [here](https://www.dropbox.com/sh/ihfbxcemxamihz9/AAD9zqnUptZzyZlquqpWllDya?dl=1). Place the models in `psla/pretrained_models/audioset/`, and run `psla/src/ensemble/ensemle.py`. You can expect similar results as follows (though for AudioSet, results can be differ as both training and evaluation data can be different).

```
# Ensemble 3 AudioSet Models Trained with Exactly Same Setting (Best Setting), But Different Random Seeds.
---------------Ensemble Result Summary---------------
Model 0 ../../pretrained_models/audioset/as_mdl_0.pth mAP: 0.440298, AUC: 0.974047, d-prime: 2.749102
Model 1 ../../pretrained_models/audioset/as_mdl_1.pth mAP: 0.439790, AUC: 0.973978, d-prime: 2.747493
Model 2 ../../pretrained_models/audioset/as_mdl_2.pth mAP: 0.439322, AUC: 0.973591, d-prime: 2.738487
Ensemble 3 Models mAP: 0.464112, AUC: 0.978222, d-prime: 2.854353

# Ensemble 5 Top-Performance AudioSet Models.
---------------Ensemble Result Summary---------------
Model 0 ../../pretrained_models/audioset/as_mdl_0.pth mAP: 0.440298, AUC: 0.974047, d-prime: 2.749102
Model 1 ../../pretrained_models/audioset/as_mdl_1.pth mAP: 0.439790, AUC: 0.973978, d-prime: 2.747493
Model 2 ../../pretrained_models/audioset/as_mdl_2.pth mAP: 0.439322, AUC: 0.973591, d-prime: 2.738487
Model 3 ../../pretrained_models/audioset/as_mdl_3.pth mAP: 0.440555, AUC: 0.973639, d-prime: 2.739613
Model 4 ../../pretrained_models/audioset/as_mdl_4.pth mAP: 0.439713, AUC: 0.973579, d-prime: 2.738213
Ensemble 5 Models mAP: 0.469050, AUC: 0.978875, d-prime: 2.872325

# Ensemble All 10 AudioSet Models Presented in the PSLA Paper
---------------Ensemble Result Summary---------------
Model 0 ../pretrained_models/audioset/as_mdl_1.pth mAP: 0.440298, AUC: 0.974047, d-prime: 2.749102
Model 1 ../pretrained_models/audioset/as_mdl_0.pth mAP: 0.439790, AUC: 0.973978, d-prime: 2.747493
Model 2 ../pretrained_models/audioset/as_mdl_2.pth mAP: 0.439322, AUC: 0.973591, d-prime: 2.738487
Model 3 ../pretrained_models/audioset/as_mdl_3.pth mAP: 0.440555, AUC: 0.973639, d-prime: 2.739613
Model 4 ../pretrained_models/audioset/as_mdl_4.pth mAP: 0.439713, AUC: 0.973579, d-prime: 2.738213
Model 5 ../pretrained_models/audioset/as_mdl_5.pth mAP: 0.438852, AUC: 0.973534, d-prime: 2.737183
Model 6 ../pretrained_models/audioset/as_mdl_6.pth mAP: 0.394262, AUC: 0.973054, d-prime: 2.726193
Model 7 ../pretrained_models/audioset/as_mdl_7.pth mAP: 0.370860, AUC: 0.961183, d-prime: 2.495504
Model 8 ../pretrained_models/audioset/as_mdl_8.pth mAP: 0.426624, AUC: 0.973353, d-prime: 2.733006
Model 9 ../pretrained_models/audioset/as_mdl_9.pth mAP: 0.372092, AUC: 0.970509, d-prime: 2.670498
Ensemble 10 Models mAP: 0.474380, AUC: 0.981043, d-prime: 2.935611
```

## Label Enhancement

Label enhancement code is in `psla/src/label_enhancement`, specifically, we have seperate code for Type-I error fixing (`fix_type1.py`) and Type-II error fixing (`fix_type2.py`). They can be combined by using `merge_type_1_2.py`. All code depends on the output of a model trained of original label set and the ground truth. We have provided ours [here](https://www.dropbox.com/s/kd84fq9ygwmidvp/prediction_files.zip?dl=1).

For sample usage, please see [FSD50K recipe](), step 3.

## Ensemble and Weight Averaging

Ensemble and weigth averaging code is in `psla/src/[ensemble,weight_averaging].py`. 

We present 3 ensemble strategy in the paper. 
 - Checkpoints of a Single Run: train the model once, but save checkpoint model every epoch. Then average the prediction of each checkpoint model.
 - Multiple Runs with Same Setting: train the model multiple times with same setting but different random seeds. Then average the prediction of the last model of each run.
 - Models Trained with Different Settings: train the model multiple times with different settings. Then average the prediction of the last model of each run.

The first is implemented in `src/traintest.py` and the second and third are implemented in `src/ensemble/ensemble.py`. We provide our pretrained ensemble model, please see the next section.

Weight averaging is averaging the model weights of last few checkpoints, it is implemented in both `src/traintest.py` and `src/ensemble/weight_averaging.py`. We provide the pretrained weight averaging model, please see the next section. Unlike ensemble, weight averaging does not increase the computational overhead (it is same with a single model), but can improve the performance.

## Pretrained Models
We provide full AudioSet and FSD50K pretrained models (click the mAP to download the model(s)).

|                                          | # Models  |AudioSet (Eval mAP) | FSD50K (Eval mAP) |
|------------------------------------------|:------:|:--------:|:------:|
| Single Model                           |  1 |[0.440](https://www.dropbox.com/s/d1z27wj30ew5qrs/as_mdl_0.pth?dl=1)  |  [0.559](https://www.dropbox.com/s/stzrmfty2oyqnnj/fsd_mdl_best_single.pth?dl=1) |
| Weight Averaging Model                 |  1 | [0.444](https://www.dropbox.com/s/ieggie0ara4x26d/as_mdl_0_wa.pth?dl=1)  |  [0.562](https://www.dropbox.com/s/5fvybrbulvhsish/fsd_mdl_wa.pth?dl=1) |
| Ensemble (Single Run, All Checkpoints) | 30/40 |[0.453](https://www.dropbox.com/sh/jo6te8fcy1ptabw/AAAtJ9sMn93-3L0XkebzQQxIa?dl=1)  |   [0.573](https://www.dropbox.com/sh/gyv95m53sib36vk/AADWCgApSxtAEVU1KrnQApi3a?dl=1)  |
| Ensemble (3 Run with Same Setting) | 3 |  [0.464](https://www.dropbox.com/sh/c83w8816vl6yhty/AADjoO9irfP1RCr-qyZMJg_-a?dl=1)  |   N/A  |
| Ensemble (All, Different Settings) | 10 |  [0.474](https://www.dropbox.com/sh/ihfbxcemxamihz9/AAD9zqnUptZzyZlquqpWllDya?dl=1)  |   N/A  |

All models are EfficientNet B2 model with 4-headed attention with 13.6M parameters, trained with 16kHz audio. Load the model by using follows:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
num_class = 527 if dataset=='audioset' else 200
audio_model = models.EffNetAttention(label_dim=num_class, b=2, pretrain=False, head_num=4)
audio_model = torch.nn.DataParallel(audio_model)
audio_model.load_state_dict(sd, strict=False)
```

We strongly recommend to use the pretrained model with our dataloader for inference to avoid the dataloading mismatch. Please see [here]().

## Pretrained Enhanced Label Set

We provide pretrained enhanced label set for AudioSet and FSD50K (click the first column to download the label sets), which can be used as drop-in replacement of original label set. In other words, no matter what model and training pipeline you are using, you can get some \`free\` performance improvement for balanced AudioSet and FSD50K by using the enhanced label set.

If you use AudioSet and FSD50K recipe, you do not need this, the recipe handles the label enhancement. If you want to generate enhanced label sets by yourself, you can skip this and just check [here]().



| Label Modification Threshold   | 5th percentile | 10th percentile | 25th percentile |  Mean  |   Median   | No (Original) |
|--------------------------------|:--------------:|:---------------:|:---------------:|:------:|:----------:|:-------------:|
| [AudioSet Full Training Set](https://www.dropbox.com/sh/0wei083a86dwgrq/AAB5ZxH9QqqRDWS-ov0_dmg3a?dl=1)     |     0.394     |      0.409     |      0.430     | 0.439 | NA |     **0.440**    |
| [AudioSet Balanced Training Set](https://www.dropbox.com/sh/hc7roh391rywn81/AABrpjoq4_VrjT-nfr5d34tJa?dl=1) |     0.308     |      0.308     |      0.317     | **0.316** | NA |     0.312    |
| [FSD50K Training Set](https://www.dropbox.com/sh/2743wk6cgnt09za/AABRNTzvvas7PF2QZjx4WPC1a?dl=1)           |   NA   |    NA   |    NA  |  **0.567** | NA |     0.558     |

All results on the above table are mAP on the **original** evaluation set (i.e., the label of the evaluation set has **NOT** been modified), training setting might not be optimal, but experiments of each row are in the same setting, numbers should be used for reference only. 

## Use Pretrained Model for Audio Tagging Inference in One Click
We do not have a plan to release an inference script for the PSLA model. But we offer an one-click inference scripe for our Audio Spectrogram Transformer (AST) model at [Google Colab](https://colab.research.google.com/github/YuanGongND/ast/blob/master/colab/AST_Inference_Demo.ipynb) (no GPU needed). The AST model is different from PSLA model, but in terms of performance, AST is stronger.

## Use PSLA Training Pipeline For New Models

You can certainly use the PSLA pipeline with your model. It is easy and can be done in one minute, just add your model to `psla/src/models/Models.py`, include it in `psla/src/run.py`, and change `model` in `run.sh` in the recipe. Your model should take input in shape `[batch_size, time_frame_num, frequency_bins]`, e.g., (12, 1056, 128) and output a tensor in shape `[batch_size, num_classes].`

You might need to search hyper-parameters for your model, but you can start with our parameters.

## Use PSLA Training Pipeline For New Datasets and Tasks

You can certainly use the PSLA pipeline with a new task or dataset. We intentionally make everything in `psla/src/run.py` and `psla/src/traintest.py` dataset-agnostic. So you just need to prepare your data in Json files and change `run.sh` in the recipe. We suggest to start with the FSD50K recipe.

## Use Pretrained CNN+Attention Model For New Tasks

You can also use our pretrained CNN+Attention model for your new task. This is extremely easy, simply first do the above step and then uncomment the code block in `psla/src/run.py`. Note, if your `num_class` is not same with the pretrained model, you will only get part of the pretrained model.

 ## Contact
If you have a question, please bring up an issue (preferred) or send me an email yuangong@mit.edu.
