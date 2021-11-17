import shutil
import sys
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
import numpy as np
import pickle
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
#from torch.cuda.amp import autocast
import ast

def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.set_grad_enabled(True)
    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    swa_sign = False
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)

    if args.resume:
        progress_pkl = "%s/progress.pkl" % exp_dir
        progress, epoch, global_step, best_epoch, best_mAP = load_progress(progress_pkl)
        print("\nResume training from:")
        print("  epoch = %s" % epoch)
        print("  global_step = %s" % global_step)
        print("  best_epoch = %s" % best_epoch)
        print("  best_mAP = %.4f" % best_mAP)

    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)

    if args.cont != None:
        audio_model.load_state_dict(torch.load(args.cont))
        print("loaded parameters from " + args.cont)

    if epoch != 0:
        audio_model.load_state_dict(torch.load("%s/models/audio_model.%d.pth" % (exp_dir, epoch)))
        print("loaded parameters from epoch %d" % epoch)

    audio_model = audio_model.to(device)
    # Set up the optimizer
    audio_trainables = [p for p in audio_model.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1000000))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_trainables) / 1000000))
    trainables = audio_trainables
    if args.optim == 'sgd':
       optimizer = torch.optim.SGD(trainables, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(trainables, args.lr,
                                weight_decay=args.weight_decay,
                                betas=(0.95, 0.999))
    else:
        raise ValueError('Optimizer %s is not supported' % args.optim)

    # LR scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)
    #if train whole set
    if len(train_loader.dataset) > 2e5:
        # print('now use scheduler 1')
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], gamma=0.5, last_epoch=-1)
        # scheduler 2 is the best choice
        # print('now use scheduler 2')
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], gamma=0.5, last_epoch=-1)
        # # print('now use scheduler 3')
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], gamma=0.8, last_epoch=-1)
        # print('now use scheduler 4')
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], gamma=0.5, last_epoch=-1)

        # #original scheduler used in the paper
        print('now use original scheduler')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25, 30, 35, 40, 45], gamma=0.5, last_epoch=-1)
    # if 200k
    elif len(train_loader.dataset) > 1.5e5:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15, 20, 25, 30, 35, 40, 45, 50, 55], gamma=0.5, last_epoch=-1)
    # if 100k
    elif len(train_loader.dataset) > 0.5e5:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20, 25, 30, 35, 40, 45, 50, 55, 60], gamma=0.5, last_epoch=-1)
    # if fsd
    elif len(train_loader.dataset) > 0.3e5:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25, 30, 35, 40, 45, 50], gamma=0.5, last_epoch=-1)
    #if balanced set
    # elif len(train_loader.dataset) > 0.15e5:
        # print('test extended scheduler')
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80, 90, 100, 110, 120], gamma=0.5, last_epoch=-1)
    elif len(train_loader.dataset) > 0.15e5:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35, 40, 45, 50, 55, 60, 65, 70, 75], gamma=0.5, last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=args.lr_patience, verbose=True)

    # swa setting
    swa_model = AveragedModel(audio_model)

    # this is in the initialization
    if epoch != 0:
        optimizer.load_state_dict(torch.load("%s/models/optim_state.%d.pth" % (exp_dir, epoch)))
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        print("loaded state dict from epoch %d" % epoch)

        # lr scheduler also need to be updated
        if len(train_loader.dataset) > 2e5:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15, 20, 25, 30, 35, 40, 45], gamma=0.5, last_epoch=epoch-1)
        # if balanced set
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35, 40, 45, 50, 55, 60, 65, 70, 75], gamma=0.5, last_epoch=epoch-1)
        # for _ in range(20):
        #     scheduler.step()
        #     print(scheduler.last_epoch, scheduler.get_last_lr())
        print(epoch)
        print('Current resume LR : ' + str(optimizer.param_groups[0]['lr']))

    epoch += 1

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    if args.resume:
        result = np.loadtxt(exp_dir + '/result.csv', delimiter=',')
    else:
        result = np.zeros([args.n_epochs + 1, 10])
    audio_model.train()
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        audio_model.train()
        print(datetime.datetime.now())

        for i, (audio_input, nframes, labels, _) in enumerate(train_loader):
            # measure data loading time
            B = audio_input.size(0)
            audio_input = audio_input.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            # if this is the first batch of the first epoch, init the encodings
            #if epoch == 0 and i == 0:
            #    audio_model.module.init_vq(audio_input)

            # currently the model models.ResNet50FullAttention
            audio_output, _ = audio_model(audio_input, nframes)

            epsilon = 1e-7
            audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)

            # then a weight needs to be loaded.
            if args.balance_class != 'none':
                #print('now using class-balanced loss')
                class_weight = torch.tensor(np.loadtxt(args.balance_class, delimiter=',')).to(device)
                num_class = len(class_weight)
                loss = - torch.mean(((labels * torch.log(audio_output)) * class_weight + ((1 - labels) * torch.log(1 - audio_output))) / (1+class_weight)) * 2

            else:
                # prediction_loss = nn.MultiLabelMarginLoss(predictions, labels)
                loss_fn = nn.BCELoss()
                #loss_fn = nn.BCEWithLogitsLoss()

                # add clamp to avoid BCE loss issue - sigmoid should be better but not working here.
                loss = loss_fn(audio_output, labels)
                #loss = - torch.mean(((labels * torch.log(audio_output)) + ((1 - labels) * torch.log(1 - audio_output))))

            # original optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if swa_sign == True:
                swa_model.update_parameters(audio_model)

            # # amp optimiztion
            # optimizer.zero_grad()
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

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
                  'Train Loss {loss_meter.val:.4f}\t'.format(
                   epoch, i, len(train_loader), per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1

        print('start validation')
        stats, valid_loss = validate(audio_model, test_loader, args, epoch)
        print('validation finished')

        # if swa_sign == True:
        #     #torch.optim.swa_utils.update_bn(train_loader, swa_model, device)
        #     #stats, valid_loss = validate(swa_model, test_loader, args, epoch)
        #     torch.save(swa_model.state_dict(), "%s/models/swa_audio_model.%d.pth" % (exp_dir, epoch))

        cum_stats = validate_ensemble(args, epoch)

        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = np.mean([stat['acc'] for stat in cum_stats])

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = np.mean([stat['acc'] for stat in stats])

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        print("mAP: {:.6f}".format(mAP))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))
        if args.esc == True:
            result[epoch, :] = [mAP, acc, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_acc, optimizer.param_groups[0]['lr']]
        else:
            result[epoch, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')

        if acc > best_acc:
            best_acc = acc
            best_acc_epoch = epoch

        if mAP > best_mAP:
            best_epoch = epoch
            best_mAP = mAP
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if args.save_model == True:
            torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
            if len(train_loader.dataset) > 2e5:
                torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        # automatic quit if no improvement in 5 epochs
        if args.esc == True:
            if acc < best_acc and epoch > (best_acc_epoch + 15):
                break
        else:
            if len(train_loader.dataset) > 2e5 and epoch > 50:
                break
            elif len(train_loader.dataset) < 2e5 and epoch > 120:
                break

        # LR schedule
        # if epoch > 5:
        scheduler.step()

        # now use swa
        if optimizer.param_groups[0]['lr'] <= args.lr/4:
            print('now using swa')
            swa_sign = True

        # # gradually remove data augmentation if performance stop improving
        # if mAP < best_mAP and epoch > (best_epoch + 1):
        #     train_loader.dataset.mixup = train_loader.dataset.mixup / 2
        #     train_loader.dataset.freqm = int(train_loader.dataset.freqm / 2)
        #     train_loader.dataset.timem = int(train_loader.dataset.timem / 2)
        #     print('now reduce augmentation {:f}, {:d}, {:d}'.format(train_loader.dataset.mixup,
        #                                                             train_loader.dataset.freqm,
        #                                                             train_loader.dataset.timem))

        print('number of params groups:' + str(len(optimizer.param_groups)))
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

def validate(audio_model, val_loader, args, epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    N_examples = val_loader.dataset.__len__()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input, nframes, labels, _) in enumerate(val_loader):
            audio_input = audio_input.to(device)

            # # Since we need to do RNN pack, we need to first sort in in descending order
            # _, indices = torch.sort(nframes, descending=True)
            #
            # audio_input = audio_input[indices]
            # nframes = nframes[indices]
            # labels = labels[indices]

            # compute output
            audio_output, _ = audio_model(audio_input, nframes)
            predictions = audio_output.to('cpu').detach()

            A_predictions.append(predictions)
            A_targets.append(labels)

            # compute the loss
            epsilon = 1e-7
            labels = labels.to(device)
            audio_output = torch.clamp(audio_output, epsilon, 1. - epsilon)
            loss_fn = nn.BCELoss()
            # loss without reduction, easy to check per-sample loss
            loss = loss_fn(audio_output, labels)
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        print("dim: ")
        print(audio_output.dim())
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats