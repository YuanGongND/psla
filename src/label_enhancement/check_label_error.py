# -*- coding: utf-8 -*-
# @Time    : 5/24/21 12:55 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : check_label_error.py

# This is an example (male, female, and kid speech classes) showing the label error in the AudioSet.

import json
"""
0,/m/09x0r,"Speech"
1,/m/05zppz,"Male speech, man speaking"
2,/m/02zsn,"Female speech, woman speaking"
3,/m/0ytgt,"Child speech, kid speaking"
"""

def check_type1_error(json_path):
    total_speech_cnt = 0
    male_cnt= 0
    female_cnt = 0
    child_cnt = 0
    with open(json_path,'r',encoding='utf8')as fp:
        data_file = json.load(fp)
        data = data_file['data']
        # for each sample
        for i, sample in enumerate(data):
            sample_labels = sample['labels'].split(',')
            if '/m/09x0r' in sample_labels:
                total_speech_cnt += 1
            if '/m/05zppz' in sample_labels:
                male_cnt += 1
            if '/m/02zsn' in sample_labels:
                female_cnt += 1
            if '/m/0ytgt' in sample_labels:
                child_cnt += 1
    print('Type-I Error:')
    print('There are {:d}, {:d}, {:d} samples that are labeled as male, female, and child speech (sum: {:d}), but there are {:d} samples labeled as speech in {:s}.'.format(male_cnt, female_cnt, child_cnt, (male_cnt+female_cnt+child_cnt), total_speech_cnt ,json_path))


def check_type2_error(json_path):
    miss_male_cnt=0
    miss_female_cnt = 0
    miss_child_cnt=0
    with open(json_path,'r',encoding='utf8')as fp:
        data_file = json.load(fp)
        data = data_file['data']
        # for each sample
        for i, sample in enumerate(data):
            sample_labels = sample['labels'].split(',')
            if '/m/05zppz' in sample_labels and '/m/09x0r' not in sample_labels:
                miss_male_cnt += 1
            if '/m/02zsn' in sample_labels and '/m/09x0r' not in sample_labels:
                miss_female_cnt += 1
            if '/m/0ytgt' in sample_labels and '/m/09x0r' not in sample_labels:
                miss_child_cnt += 1
    print('Type-II Error:')
    print('There are {:d}, {:d}, {:d} samples that are labeled as male, female, and child speech, respectively, but are not labeled as speech in {:s}.'.format(miss_male_cnt, miss_female_cnt, miss_child_cnt, json_path))

# before label enhancement
check_type1_error('../../egs/audioset/datafiles/balanced_train_data.json')
check_type2_error('../../egs/audioset/datafiles/balanced_train_data.json')

# after label enhancement
check_type1_error('../../egs/audioset/datafiles/balanced_train_data_type1_2_mean.json')
check_type2_error('../../egs/audioset/datafiles/balanced_train_data_type1_2_mean.json')