# -*- coding: utf-8 -*-
# @Time    : 6/10/21 11:00 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : traintest.py

import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_ensemble_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_ensemble_epoch, best_mAP, best_acc, best_ensemble_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP, time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))

    # dataset specific settings
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(args.lrscheduler_start, 1000, 5)), gamma=args.lrscheduler_decay, last_epoch=epoch - 1)
    main_metrics = args.metrics
    if args.loss == 'BCE':
        loss_fn = nn.BCELoss()
    elif args.loss == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    warmup = args.warmup
    args.loss_fn = loss_fn
    print('now training with {:s}, main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(args.dataset), str(main_metrics), str(loss_fn), str(scheduler)))
    print('The learning rate scheduler starts at {:d} epoch with decay rate of {:.3f} '.format(args.lrscheduler_start, args.lrscheduler_decay))

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input, labels) in enumerate(train_loader):

            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            audio_output = audio_model(audio_input)
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(audio_output, torch.argmax(labels.long(), axis=1))
            else:
                epsilon = 1e-7
                audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
                loss = loss_fn(audio_output, labels)

            # optimization if amp is not used
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)

        # ensemble results
        ensemble_stats = validate_ensemble(args, epoch)
        ensemble_mAP = np.mean([stat['AP'] for stat in ensemble_stats])
        ensemble_mAUC = np.mean([stat['auc'] for stat in ensemble_stats])
        ensemble_acc = ensemble_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, ensemble_mAP, ensemble_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, ensemble_acc, ensemble_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if ensemble_mAP > best_ensemble_mAP:
            best_ensemble_epoch = epoch
            best_ensemble_mAP = ensemble_mAP

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        if len(train_loader.dataset) > 2e5:
            torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()

    # if test weight averaging
    if args.wa == True:
        stats=validate_wa(audio_model, test_loader, args, args.wa_start, args.wa_end)
        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)
        wa_result = [mAP, mAUC]
        print('---------------Training Finished---------------')
        # print('On Validation Set')
        # print('weighted averaged model results')
        # print("mAP: {:.6f}".format(mAP))
        # print("AUC: {:.6f}".format(mAUC))
        # print("d_prime: {:.6f}".format(d_prime(mAUC)))
        np.savetxt(exp_dir + '/wa_result.csv', wa_result)

def validate(audio_model, val_loader, args, epoch, eval_target=False):  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):    
        audio_model = nn.DataParallel(audio_model)  
    audio_model = audio_model.to(device)    
    # switch to evaluate mode   
    audio_model.eval()  
    end = time.time()   
    A_predictions = []  
    A_targets = []  
    A_loss = [] 
    with torch.no_grad():   
        for i, (audio_input, labels) in enumerate(val_loader):  
            audio_input = audio_input.to(device)    
            # compute output    
            audio_output = audio_model(audio_input) 
            predictions = audio_output.to('cpu').detach()   
            A_predictions.append(predictions)   
            A_targets.append(labels)    
            # compute the loss  
            labels = labels.to(device)  
            epsilon = 1e-7  
            audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon) 
            if isinstance(args.loss_fn, torch.nn.CrossEntropyLoss): 
                loss = args.loss_fn(audio_output, torch.argmax(labels.long(), axis=1))  
            else:   
                loss = args.loss_fn(audio_output, labels)   
            A_loss.append(loss.to('cpu').detach())  
            batch_time.update(time.time() - end)    
            end = time.time()   
        audio_output = torch.cat(A_predictions) 
        target = torch.cat(A_targets)   
        loss = np.mean(A_loss)  
        stats = calculate_stats(audio_output, target)   
        # save the prediction here  
        exp_dir = args.exp_dir  
        if os.path.exists(exp_dir+'/predictions') == False: 
            os.mkdir(exp_dir+'/predictions')    
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')    
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')  
        # save the target for the separate eval set if there's one. 
        if eval_target == True and os.path.exists(exp_dir+'/predictions/eval_target.csv') == False: 
            np.savetxt(exp_dir + '/predictions/eval_target.csv', target, delimiter=',') 
    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        ensemble_predictions = np.loadtxt(exp_dir + '/predictions/ensemble_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        ensemble_predictions = ensemble_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    ensemble_predictions = ensemble_predictions / epoch
    np.savetxt(exp_dir+'/predictions/ensemble_predictions.csv', ensemble_predictions, delimiter=',')

    stats = calculate_stats(ensemble_predictions, target)
    return stats

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')

    stats, loss = validate(audio_model, val_loader, args, 'wa')
    return stats