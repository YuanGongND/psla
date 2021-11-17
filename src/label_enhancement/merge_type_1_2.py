# -*- coding: utf-8 -*-
# @Time    : 12/24/20 2:34 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : merge_type_1_2.py

# merge label enhancement 1 and label enhancement 2.

import json
import os

# count the total number of labels (one audio clip may have multiple labels) of a data json file.
def count_label(js):
    cnt = 0
    with open(js,'r',encoding='utf8')as fp:
        data_file = json.load(fp)
        data = data_file['data']
        for i, sample in enumerate(data):
            sample_labels = sample['labels'].split(',')
            cnt += len(sample_labels)
    return cnt

# merge datafiles of type-1 label enhancement and type-2 label enhancement.
def merge_label(js1, js2, output_path):
    total_label_cnt = 0
    with open(js1,'r',encoding='utf8')as fp:
        data_file = json.load(fp)
        data1 = data_file['data']
    with open(js2, 'r', encoding='utf8')as fp:
        data_file = json.load(fp)
        data2 = data_file['data']
    for i, sample in enumerate(data1):
        sample_labels1 = sample['labels'].split(',')
        sample_labels2 = data2[i]['labels'].split(',')
        merge_label = list(set(sample_labels1 + sample_labels2))
        data1[i]['labels'] = ','.join(list(set(merge_label)))
        total_label_cnt += len(list(set(merge_label)))
    output = {'data': data1}
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=1)
    print('Input Json file 1 has {:d} labels'.format(count_label(js1)))
    print('Input Json file 2 has {:d} labels'.format(count_label(js2)))
    print('Merged Json file has {:d} labels'.format(total_label_cnt))

if __name__ == '__main__':
    # 'audioset' or 'fsd50k'
    # for audioset, we demo label enhancement on the balanced training set
    dataset = 'fsd50k'
    # for different label modification threshold
    for p in ['mean', 'median', '25', '10', '5']:
        print('----------------Merge Type 1&2 Label Enhancement with {:s} Threshold'.format(p))
        if dataset == 'fsd50k':
            path1 = '../../egs/' + dataset + '/datafiles/fsd50k_tr_full_type1_' + p + '.json'
            path2 = '../../egs/' + dataset + '/datafiles/fsd50k_tr_full_type2_' + p + '.json'
            out_path = '../../egs/' + dataset + '/datafiles/fsd50k_tr_full_type1_2_' + p + '.json'
            merge_label(path1, path2, out_path)

        if dataset == 'audioset':
            path1 = '../../egs/' + dataset + '/datafiles/balanced_train_data_type1_' + p + '.json'
            path2 = '../../egs/' + dataset + '/datafiles/balanced_train_data_type2_' + p + '.json'
            out_path = '../../egs/' + dataset + '/datafiles/balanced_train_data_type1_2_' + p + '.json'
            merge_label(path1, path2, out_path)

        # (optional) generate balanced sampling weight for each enhanced label set.
        os.system('python ../gen_weight_file.py --dataset {:s} --label_indices_path {:s} --datafile_path {:s}'.format(
                dataset, '../../egs/' + dataset + '/class_labels_indices.csv', out_path))
