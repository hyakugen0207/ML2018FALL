import pandas as pd
import numpy as np
import sys
import os


input_path = sys.argv[1]
output_path = sys.argv[2]


def get_new_test_data(data):
    new_data = np.zeros((2340, 18))
    count = 0
    for raw in range(0, len(data), 18):
        for col in range(2, 11):
            for feature in range(raw, raw+18):
                if count < len(data) and data[feature][col] == 'NR':
                    new_data[count][feature-raw] = 0.0
                else:
                    if count < len(data):
                        new_data[count][feature-raw] = data[feature][col]
            count += 1

    cut_raw_count = 0
    cut_data = np.zeros((260, 18*9))
    for raw in range(0, len(new_data)-8, 9):
            a = np.array([])
            cut_data[cut_raw_count] = np.append(a, new_data[raw:raw+9])
            cut_raw_count += 1
    return cut_data


#read test data
test_data_for_id = pd.read_csv(input_path, encoding='big5', names=['id', 'attr'] + [str(r) for r in range(9)]).id.unique()
test_data = pd.read_csv(input_path, encoding='big5', names=['id', 'attr'] + [str(r) for r in range(9)]).values
new_test_data = get_new_test_data(test_data)

#read weight
weight_path1 = './best_w_[0928-Kfold+shuffle+0928best_w_bias-ver7].csv'
weight_path2 = './best_w_[0930-mean[[0928-Kfold+shuffle+0928best_w_bias-ver7]+[0930-Kfold+shuffle+new_feature2-ver10]]-ver11].csv'
w1 = pd.read_csv(weight_path1, encoding='big5').values
w2 = pd.read_csv(weight_path2, encoding='big5').values

best_w = np.zeros(162)
best_bias = 0
for i in range(0,162):
    best_w[i]=(w1[i]+w2[i])/2

best_bias = (w1[162]+w2[162])/2


# output public testing predict file


ans_value = np.dot(new_test_data, best_w)+best_bias
ans = pd.DataFrame()
ans = ans.assign(id=test_data_for_id)
ans = ans.assign(value=ans_value)
ans.to_csv(output_path, index=False)