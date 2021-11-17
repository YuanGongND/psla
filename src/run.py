# Yuan Gong, modified from:
# Author: David Harwath
import argparse
import os
import pickle
import sys
from collections import OrderedDict
import time
import torch
import shutil
basepath = os.path.dirname(os.path.dirname(sys.path[0]))
sys.path.append(basepath)
import dataloaders
from utilities import *
import models
from traintest import train, validate
import ast
from torch.utils.data import WeightedRandomSampler
import numpy as np

print("I am process %s, running on %s: starting (%s)" % (
        os.getpid(), os.uname()[1], time.asctime()))

# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data-train", type=str, default='', help="training data json")
parser.add_argument("--data-val", type=str, default='', help="validation data json")
parser.add_argument("--data-eval", type=str, default=None, help="evaluation data json")
parser.add_argument("--label-csv", type=str, default=os.path.join(basepath, 'utilities/class_labels_indices_coarse.csv'), help="csv with class labels")
parser.add_argument("--exp-dir", type=str, default="", help="directory to dump experiments")

# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=60, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('-w', '--num-workers', default=8, type=int, metavar='NW', help='# of workers for dataloading (default: 8)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--lr-decay', default=40, type=int, metavar='LRDECAY', help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-7, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--n-epochs", type=int, default=1, help="number of maximum training epochs")
parser.add_argument("--n-print-steps", type=int, default=1, help="number of steps to print statistics")

# model args
parser.add_argument("--model", type=str, default="efficientnet", help="audio model architecture", choices=["efficientnet", "resnet", "mbnet"])
parser.add_argument("--dataset", type=str, default="audioset", help="the dataset used", choices=["audioset", "esc50", "speechcommands"])

parser.add_argument("--dataset_mean", type=float, default=-4.6476, help="the dataset mean, used for input normalization")
parser.add_argument("--dataset_std", type=float, default=4.5699, help="the dataset std, used for input normalization")
parser.add_argument("--target_length", type=int, default=1056, help="the input length in frames")
parser.add_argument("--noise", help='if use balance sampling', type=ast.literal_eval)
parser.add_argument("--metrics", type=str, default="mAP", help="the main evaluation metrics", choices=["mAP", "acc"])
parser.add_argument("--warmup", help='if use balance sampling', type=ast.literal_eval)
parser.add_argument("--loss", type=str, default="BCE", help="the loss function", choices=["BCE", "CE"])
parser.add_argument("--lrscheduler_start", type=int, default=10, help="when to start decay")
parser.add_argument("--lrscheduler_decay", type=float, default=0.5, help="the learning rate decay ratio")
parser.add_argument("--wa", help='if do weight averaging', type=ast.literal_eval)
parser.add_argument("--wa_start", type=int, default=16, help="which epoch to start weight averaging")
parser.add_argument("--wa_end", type=int, default=30, help="which epoch to end weight averaging")

parser.add_argument("--n_class", type=int, default=527, help="number of classes")
parser.add_argument('--save_model', help='save the model or not', type=ast.literal_eval)
parser.add_argument("--eff_b", type=int, default=0, help="which efficientnet to use, the larger number, the more complex")
parser.add_argument('--esc', help='If doing an ESC exp, which will have some different behabvior', type=ast.literal_eval, default='False')
parser.add_argument('--impretrain', help='if use imagenet pretrained CNNs', type=ast.literal_eval, default='True')
parser.add_argument('--freqm', help='frequency mask max length', type=int, default=0)
parser.add_argument('--timem', help='time mask max length', type=int, default=0)
parser.add_argument("--mixup", type=float, default=0, help="how many (0-1) samples need to be mixup during training")
parser.add_argument("--lr_patience", type=int, default=2, help="how many epoch to wait to reduce lr if mAP doesn't improve")
parser.add_argument("--att_head", type=int, default=4, help="number of attention heads")
parser.add_argument('--bal', help='if use balance sampling', type=ast.literal_eval)

args = parser.parse_args()

audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': args.freqm,
              'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train',
              'mean': args.dataset_mean, 'std': args.dataset_std,
              'noise': False}
val_audio_conf = {'num_mel_bins': 128, 'target_length': args.target_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                  'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean,
                  'std': args.dataset_std, 'noise': False}

if args.bal == True:
    print('balanced sampler is being used')
    samples_weight = np.loadtxt(args.data_train[:-5] + '_weight.csv', delimiter=',')
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    train_loader = torch.utils.data.DataLoader(
        dataloaders.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=False)
else:
    print('balanced sampler is not used')
    train_loader = torch.utils.data.DataLoader(
        dataloaders.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)

val_loader = torch.utils.data.DataLoader(
    dataloaders.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
    batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.data_eval != None:
    eval_loader = torch.utils.data.DataLoader(
        dataloaders.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model == 'efficientnet':
    audio_model = models.EffNetAttention(label_dim=args.n_class, b=args.eff_b, pretrain=args.impretrain, head_num=args.att_head)
elif args.model == 'resnet':
    audio_model = models.ResNetAttention(label_dim=args.n_class, pretrain=args.impretrain)
elif args.model == 'mbnet':
    audio_model = models.MBNet(label_dim=args.n_class, pretrain=args.effpretrain)

# if you want to use a pretrained model for fine-tuning, uncomment here.
# if not isinstance(audio_model, nn.DataParallel):
#     audio_model = nn.DataParallel(audio_model)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# sd = torch.load('../pretrained_models/as_mdl_0.pth', map_location=device)
# audio_model.load_state_dict(sd, strict=False)

if not bool(args.exp_dir):
    print("exp_dir not specified, automatically naming one...")
    args.exp_dir = "exp/Data-%s/AudioModel-%s_Optim-%s_LR-%s_Epochs-%s" % (
        os.path.basename(args.data_train), args.model, args.optim,
        args.lr, args.n_epochs)

print("\nCreating experiment directory: %s" % args.exp_dir)
if os.path.exists("%s/models" % args.exp_dir) == False:
    os.makedirs("%s/models" % args.exp_dir)
with open("%s/args.pkl" % args.exp_dir, "wb") as f:
    pickle.dump(args, f)

train(audio_model, train_loader, val_loader, args)

# if the dataset has a seperate evaluation set (e.g., FSD50K), then select the model using the validation set and eval on the evaluation set.
print('---------------Result Summary---------------')
if args.data_eval != None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # evaluate best single model
    sd = torch.load(args.exp_dir + '/models/best_audio_model.pth', map_location=device)
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)
    print('---------------evaluate best single model on the validation set---------------')
    stats, _ = validate(audio_model, val_loader, args, 'best_single_valid_set')
    val_mAP = np.mean([stat['AP'] for stat in stats])
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(val_mAP))
    print("AUC: {:.6f}".format(val_mAUC))
    print('---------------evaluate best single model on the evaluation set---------------')
    stats, _ = validate(audio_model, eval_loader, args, 'best_single_eval_set', eval_target=True)
    eval_mAP = np.mean([stat['AP'] for stat in stats])
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(eval_mAP))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/best_single_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

    # evaluate weight average model
    sd = torch.load(args.exp_dir + '/models/audio_model_wa.pth', map_location=device)
    audio_model.load_state_dict(sd)
    print('---------------evaluate weight average model on the validation set---------------')
    stats, _ = validate(audio_model, val_loader, args, 'wa_valid_set')
    val_mAP = np.mean([stat['AP'] for stat in stats])
    val_mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(val_mAP))
    print("AUC: {:.6f}".format(val_mAUC))
    print('---------------evaluate weight averages model on the evaluation set---------------')
    stats, _ = validate(audio_model, eval_loader, args, 'wa_eval_set')
    eval_mAP = np.mean([stat['AP'] for stat in stats])
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(eval_mAP))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/wa_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

    # evaluate the ensemble results
    print('---------------evaluate ensemble model on the validation set---------------')
    # this is already done in the training process, only need to load
    result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
    val_mAP = result[-1, -3]
    val_mAUC = result[-1, -2]
    print("mAP: {:.6f}".format(val_mAP))
    print("AUC: {:.6f}".format(val_mAUC))
    print('---------------evaluate ensemble model on the evaluation set---------------')
    # get the prediction of each checkpoint model
    for epoch in range(1, args.n_epochs+1):
        sd = torch.load(args.exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        audio_model.load_state_dict(sd)
        validate(audio_model, eval_loader, args, 'eval_'+str(epoch))
    # average the checkpoint prediction and calculate the results
    target = np.loadtxt(args.exp_dir + '/predictions/eval_target.csv', delimiter=',')
    ensemble_predictions = np.zeros_like(target)
    for epoch in range(1, args.n_epochs + 1):
        cur_pred = np.loadtxt(args.exp_dir + '/predictions/predictions_eval_' + str(epoch) + '.csv', delimiter=',')
        ensemble_predictions += cur_pred
    ensemble_predictions = ensemble_predictions / args.n_epochs
    stats = calculate_stats(ensemble_predictions, target)
    eval_mAP = np.mean([stat['AP'] for stat in stats])
    eval_mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {:.6f}".format(eval_mAP))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/ensemble_result.csv', [val_mAP, val_mAUC, eval_mAP, eval_mAUC])

# if the dataset only has evaluation set (no validation set), e.g., AudioSet
else:
    # evaluate single model
    print('---------------evaluate best single model on the evaluation set---------------')
    # result is the performance of each epoch, we average the results of the last 5 epochs
    result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
    last_five_epoch_mean = np.mean(result[-5: , :], axis=0)
    eval_mAP = last_five_epoch_mean[0]
    eval_mAUC = last_five_epoch_mean[1]
    print("mAP: {:.6f}".format(eval_mAP))
    print("AUC: {:.6f}".format(eval_mAUC))
    np.savetxt(args.exp_dir + '/best_single_result.csv', [eval_mAP, eval_mAUC])

    # evaluate weight average model
    print('---------------evaluate weight average model on the evaluation set---------------')
    # already done in training process, only need to load
    result = np.loadtxt(args.exp_dir + '/wa_result.csv', delimiter=',')
    wa_mAP = result[0]
    wa_mAUC = result[1]
    print("mAP: {:.6f}".format(wa_mAP))
    print("AUC: {:.6f}".format(wa_mAUC))
    np.savetxt(args.exp_dir + '/wa_result.csv', [wa_mAP, wa_mAUC])

    # evaluate ensemble
    print('---------------evaluate ensemble model on the evaluation set---------------')
    # already done in training process, only need to load
    result = np.loadtxt(args.exp_dir + '/result.csv', delimiter=',')
    ensemble_mAP = result[-1, -3]
    ensemble_mAUC = result[-1, -2]
    print("mAP: {:.6f}".format(ensemble_mAP))
    print("AUC: {:.6f}".format(ensemble_mAUC))
    np.savetxt(args.exp_dir + '/ensemble_result.csv', [ensemble_mAP, ensemble_mAUC])