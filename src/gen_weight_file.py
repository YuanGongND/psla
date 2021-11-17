# -*- coding: utf-8 -*-
# @Time    : 11/17/20 3:22 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : gen_weight_file.py

# gen sample weight = sum(label_weight) for label in all labels of the audio clip, where label_weight is the reciprocal of the total sample count of that class.
# Note audioset and fsd50k are multi-label datasets

import argparse
import json
import numpy as np
import sys, os, csv

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", type=str, default="audioset", help="training optimizer", choices=["audioset", "fsd50k"])
parser.add_argument("--label_indices_path", type=str, default="./class_labels_indices.csv", help="the label vocabulary file.")
parser.add_argument("--datafile_path", type=str, default='./datafiles/balanced_train_data.json', help="the path of data json file")

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup

if __name__ == '__main__':
    args = parser.parse_args()
    data_path = args.datafile_path

    index_dict = make_index_dict(args.label_indices_path)
    num_class = 527 if args.dataset == 'audioset' else 200
    label_count = np.zeros(num_class)

    with open(data_path, 'r', encoding='utf8')as fp:
        data = json.load(fp)
        data = data['data']

    for sample in data:
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            label_count[label_idx] = label_count[label_idx] + 1

    # the reason not using 1 is to avoid underflow for majority classes, add small value to avoid underflow
    label_weight = 1000.0 / (label_count + 0.01)
    sample_weight = np.zeros(len(data))

    for i, sample in enumerate(data):
        sample_labels = sample['labels'].split(',')
        for label in sample_labels:
            label_idx = int(index_dict[label])
            # summing up the weight of all appeared classes in the sample, note audioset is multiple-label classification
            sample_weight[i] += label_weight[label_idx]
    np.savetxt(data_path[:-5]+'_weight.csv', sample_weight, delimiter=',')



