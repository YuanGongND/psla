# -*- coding: utf-8 -*-
# @Time    : 12/23/20 2:02 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : fix_type1.py

# enhance the label based on the model prediction, fixing the TYPE-I error.
# TYPE-I error: an audio clip is labeled with a parent class,
# but not also labeled as a child class when it does in fact
# contain the audio event of the child class.

import json
import os

import numpy as np

def generate_child_dict():
    with open('../utilities/ontology.json', 'r', encoding='utf8')as fp:
        ontology = json.load(fp)
    # map each class to its children classes
    child_dict = {}
    for audio_class in ontology:
        cur_id = audio_class['id']
        # avoid abstract and discountinued class
        cur_restriction = audio_class['restrictions']
        if cur_restriction != ['abstract']:
            child_dict[cur_id] = audio_class['child_ids']
    return child_dict

def enhance_label_type1(json_path, output_path, child_dict, labels_code_list, score_threshold, pred, dataset='audioset'):
    num_class = 527 if dataset == 'audioset' else 200
    original_label_num, fixed_label_num, fix_case_num = 0, 0, 0
    # these are just to track the change for analysis
    child_case_cnt, par_case_cnt, class_sample_cnt = [0] * num_class, [0] * num_class, [0] * num_class
    par_child_dict = {}
    with open(json_path, 'r', encoding='utf8')as fp:
        data_file = json.load(fp)
        data = data_file['data']
        # for each audio sample
        for i, sample in enumerate(data):
            sample_labels = sample['labels'].split(',')
            new_labels = sample_labels.copy()
            original_label_num += len(sample_labels)
            # fpr each label of the audio sample
            for label in sample_labels:
                class_sample_cnt[code2idx[label]] += 1
                # there are some FSD50K classes not included in the AudioSet ontology, ingore them
                if label not in ['/m/09l8g', '/m/0bm0k', '/t/dd00012', '/m/09hlz4', '/t/dd00071'] or dataset == 'audioset':
                    # if this label has child class(es)
                    if child_dict[label] != None:
                        # one label might have multiple child classes
                        for child_label in child_dict[label]:
                            #if the child class is in 527-class list (i.e., not abstract, not discontinued, etc)
                            if child_label in labels_code_list:
                                # if the child label not already in the original label set
                                if child_label not in new_labels:
                                    # get the index of the child class
                                    child_label_idx = code2idx[child_label]
                                    # the model prediction score on the child class of this sample
                                    pred_score = pred[i, child_label_idx]
                                    # if the prediction score is higher than the threshold
                                    if pred_score > score_threshold[child_label_idx]:
                                        # add the child label
                                        new_labels.append(child_label)
                                        # below are just to track the change for analysis
                                        fix_case_num += 1
                                        par_case_cnt[code2idx[label]] += 1
                                        child_case_cnt[code2idx[child_label]] += 1
                                        if str(code2idx[label]) + '_' + str(code2idx[child_label]) not in par_child_dict:
                                            par_child_dict[str(code2idx[label]) + '_' + str(code2idx[child_label])] = 1
                                        else:
                                            par_child_dict[str(code2idx[label]) + '_' + str(code2idx[child_label])] += 1
            # remove repeated labels and add the new labels to the dataset
            data[i]['labels'] = ','.join(list(set(new_labels)))
            fixed_label_num += len(list(set(new_labels)))
    output = {'data': data}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=1)
    print('Added {:d} ({:.1f}%) labels to original {:d} original labels'.format((fixed_label_num-original_label_num), (fixed_label_num / original_label_num-1)*100, original_label_num))
    return child_case_cnt, par_case_cnt, par_child_dict, class_sample_cnt

if __name__ == '__main__':
    # 'audioset' or 'fsd50k'
    # for audioset, we demo label enhancement on the balanced training set
    dataset = 'fsd50k'
    num_class = 527 if dataset == 'audioset' else 200

    # generate a dict that maps each class to its children classes
    child_dict = generate_child_dict()

    # generate a dict that maps label code to index
    with open('../../egs/' + dataset + '/class_labels_indices.csv') as f:
        labels = f.readlines()

    labels = labels[1:]
    labels_code_list = [label.strip('\n').split(',')[1] for label in labels]
    code2idx = {labels[i].strip('\n').split(',')[1]: i for i in range(len(labels))}

    # the label enhancement algorithm depends on the soft model output prediction and ground truth
    if dataset == 'fsd50k':
        target_path = "./predictions_fsd/target.csv"
        pred_path = "./predictions_fsd/predictions.csv"
    else:
        target_path = "./predictions_as_bal/target.csv"
        pred_path = "./predictions_as_bal/predictions.csv"
    target = np.loadtxt(target_path, delimiter=',')
    pred = np.loadtxt(pred_path, delimiter=',')

    # generate label modification thresholds for each class
    mean_score = [np.mean(pred[np.where(target[:, i] == 1)[0], i]) for i in range(num_class)]
    median_score = [np.median(pred[np.where(target[:, i] == 1)[0], i]) for i in range(num_class)]
    twentyfivepercentile = [np.percentile(pred[np.where(target[:, i] == 1)[0], i], 25) for i in range(num_class)]
    tenpercentile = [np.percentile(pred[np.where(target[:, i] == 1)[0], i], 10) for i in range(num_class)]
    fivepercentile = [np.percentile(pred[np.where(target[:, i] == 1)[0], i], 5) for i in range(num_class)]

    thres_dict = {'mean': mean_score, 'median': median_score, '25': twentyfivepercentile, '10': tenpercentile, '5': fivepercentile}
    #
    for p in ['mean', 'median', '25', '10', '5']:
        threshold = thres_dict[p]
        if dataset == 'fsd50k':
            original_datafile = "../../egs/fsd50k/datafiles/fsd50k_tr_full.json"
            enhanced_datafile = "../../egs/fsd50k/datafiles/fsd50k_tr_full_type1_"+ p +".json"
        elif dataset == 'audioset':
            original_datafile = "../../egs/audioset/datafiles/balanced_train_data.json"
            enhanced_datafile = "../../egs/audioset/datafiles/balanced_train_data_type1_"+ p +".json"

        enhance_label_type1(original_datafile, enhanced_datafile, child_dict, labels_code_list, threshold, pred, dataset=dataset)
        # (optional) generate balanced sampling weight for each enhanced label set.
        os.system('python ../gen_weight_file.py --dataset {:s} --label_indices_path {:s} --datafile_path {:s}'.format(dataset, '../../egs/' + dataset + '/class_labels_indices.csv', enhanced_datafile))