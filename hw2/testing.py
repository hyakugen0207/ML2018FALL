import pandas as pd
import numpy as np
import os
import sys


train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
output_path = sys.argv[4]
#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
feature = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6
normalize = True

train_data = pd.read_csv(train_x_path, encoding='big5', dtype='float')
train_label = pd.read_csv(train_y_path, encoding='big5', dtype='float64')
test_data = pd.read_csv(test_x_path, encoding='big5', dtype='float64')
test_data_for_id = ['id_'+str(i) for i in range(0,10000)]
# modify feature of PAY_X
for var in train_data.columns[5:11]:
    train_data[var + '_no_use'] = np.where(train_data[var] == -2, 1, 0)
    train_data[var + '_payoff'] = np.where(train_data[var] == -1, 1, 0)
    train_data[var] = np.where(train_data[var] < 0, 0, train_data[var])
    test_data[var + '_no_use'] = np.where(test_data[var] == -2, 1, 0)
    test_data[var + '_payoff'] = np.where(test_data[var] == -1, 1, 0)
    test_data[var] = np.where(test_data[var] < 0, 0, test_data[var])


np_train_data = train_data[feature].values
np_label_data = train_label.values
np_test_data = test_data[feature].values


# In[5]:


#normalize
max_value = np.max(np_train_data, axis=0)
min_value = np.min(np_train_data, axis=0)
max_test_value = np.max(np_test_data, axis=0)
min_test_value = np.min(np_test_data, axis=0)

new_train_data = None
new_test_data = None

if normalize is True:
    new_train_data = (np_train_data - min_value)/(max_value - min_value)
    new_test_data = (np_test_data - min_test_value)/(max_test_value - min_test_value)
else:
    new_train_data = np_train_data
    new_test_data = np_test_data


w = [1.21045649, 1.10556599, 1.00940544, 0.99326588, 1.01001423, 1.00654496, 0.04569375,-0.00832325,-0.04625145,-0.01597579,-0.00279266,-0.03135514,-0.02640485,-0.11077803,-0.10215492,-0.0789127 ,-0.0464507 ,-0.05985495]

bias = -1.43892282


def predict(w,b,x):
    return 1/(1+np.exp(-1*(np.dot(x,w)+b)))


predict_result = np.zeros((len(new_test_data),1), dtype='int')-1
for i in range(0,len(new_test_data)):
    if predict(w,bias,new_test_data[i]) >= 0.5:
        predict_result[i][0] = 1
    else:
        predict_result[i][0] = 0

ans = pd.DataFrame()
ans = ans.assign(id=test_data_for_id)
ans = ans.assign(Value=predict_result)
print('done')
ans.to_csv(output_path,index=False)