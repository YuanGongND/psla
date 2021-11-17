# -*- coding: utf-8 -*-
# @Time    : 6/23/21 5:38 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : ensemble.py

# get the ensemble result

import os, sys, argparse
parentdir = str(os.path.abspath(os.path.join(__file__ ,"../..")))
sys.path.append(parentdir)
import dataloaders
import models
from utilities import *
from traintest import validate
import numpy as np
from scipy import stats
import torch

def get_ensemble_res(mdl_list, base_path, dataset='audioset'):
    num_class = 527 if dataset=='audioset' else 200
    # the 0-len(mdl_list) rows record the results of single models, the last row record the result of the ensemble model.
    ensemble_res = np.zeros([len(mdl_list) + 1, 3])
    if os.path.exists(base_path) == False:
        os.mkdir(base_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for model_idx, mdl in enumerate(mdl_list):
        print('-----------------------')
        print('now loading model {:d}: {:s}'.format(model_idx, mdl))

        # sd = torch.load('/Users/yuan/Documents/ast/pretrained_models/audio_model_wa.pth', map_location=device)
        sd = torch.load(mdl, map_location=device)
        if 'module.effnet._fc.weight' in sd.keys():
            del sd['module.effnet._fc.weight']
            del sd['module.effnet._fc.bias']
            torch.save(sd, mdl)
        audio_model = models.EffNetAttention(label_dim=num_class , b=2, pretrain=False, head_num=4)
        audio_model = torch.nn.DataParallel(audio_model)
        audio_model.load_state_dict(sd, strict=True)

        args.exp_dir = base_path

        stats, _ = validate(audio_model, eval_loader, args, model_idx)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        dprime = d_prime(mAUC)
        ensemble_res[model_idx, :] = [mAP, mAUC, dprime]
        print("Model {:d} {:s} mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(model_idx, mdl, mAP, mAUC, dprime))

    # calculate the ensemble result
    # get the ground truth label
    target = np.loadtxt(base_path + '/predictions/target.csv', delimiter=',')
    # get the ground truth label
    prediction_sample = np.loadtxt(base_path + '/predictions/predictions_0.csv', delimiter=',')
    # allocate memory space for the ensemble prediction
    predictions_table = np.zeros([len(mdl_list) , prediction_sample.shape[0], prediction_sample.shape[1]])
    for model_idx in range(0, len(mdl_list)):
        predictions_table[model_idx, :, :] = np.loadtxt(base_path + '/predictions/predictions_' + str(model_idx) + '.csv', delimiter=',')
        model_idx += 1

    ensemble_predictions = np.mean(predictions_table, axis=0)
    stats = calculate_stats(ensemble_predictions, target)
    ensemble_mAP = np.mean([stat['AP'] for stat in stats])
    ensemble_mAUC = np.mean([stat['auc'] for stat in stats])
    ensemble_dprime = d_prime(ensemble_mAUC)
    ensemble_res[-1, :] = [ensemble_mAP, ensemble_mAUC, ensemble_dprime]
    print('---------------Ensemble Result Summary---------------')
    for model_idx in range(len(mdl_list)):
        print("Model {:d} {:s} mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(model_idx, mdl_list[model_idx], ensemble_res[model_idx, 0], ensemble_res[model_idx, 1], ensemble_res[model_idx, 2]))
    print("Ensemble {:d} Models mAP: {:.6f}, AUC: {:.6f}, d-prime: {:.6f}".format(len(mdl_list), ensemble_mAP, ensemble_mAUC, ensemble_dprime))
    np.savetxt(base_path + '/ensemble_result.csv', ensemble_res, delimiter=',')

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

# dataloader settings
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
args = parser.parse_args()

dataset = 'audioset'
# uncomment this line if you want test ensemble on fsd50k
# dataset = 'fsd50k'

args.dataset = dataset
if dataset == 'audioset':
    args.data_eval = '../../egs/audioset/datafiles/eval_data.json'
else:
    args.data_eval = '../../egs/fsd50k/datafiles/fsd50k_eval_full.json'
args.label_csv='../../egs/' + dataset + '/class_labels_indices.csv'
args.loss_fn = torch.nn.BCELoss()
norm_stats = {'audioset': [-4.6476, 4.5699], 'fsd50k': [-4.6476, 4.5699]}
target_length = {'audioset': 1056, 'fsd50k': 3000}
batch_size = 200 if dataset=='audioset' else 48

val_audio_conf = {'num_mel_bins': 128, 'target_length': target_length[args.dataset], 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':norm_stats[args.dataset][0], 'std':norm_stats[args.dataset][1], 'noise': False}
eval_loader = torch.utils.data.DataLoader(
    dataloaders.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

if dataset == 'audioset':
    # ensemble 3 audioset models trained with exactly same setting, but different random seeds
    mdl_list_3 = ['../../pretrained_models/audioset/as_mdl_'+str(i)+'.pth' for i in range(3)]

    # ensemble top 5 audioset models, mAP =
    mdl_list_5 = ['../../pretrained_models/audioset/as_mdl_'+str(i)+'.pth' for i in range(5)]

    # ensemble entire 10 audioset models, mAP =
    mdl_list_10 = ['../../pretrained_models/audioset/as_mdl_'+str(i)+'.pth' for i in range(10)]

    get_ensemble_res(mdl_list_3, './ensemble_as', dataset)
    get_ensemble_res(mdl_list_5, './ensemble_as', dataset)
    get_ensemble_res(mdl_list_10, './ensemble_as', dataset)

else:
    pass