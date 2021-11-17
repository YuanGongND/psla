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

If you are experimenting with Balanced AudioSet or want to enhance the labelset anyway, check our [pretrained enhanced label set](here), and change the labels in your datafile.

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