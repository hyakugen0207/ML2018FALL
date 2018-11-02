import pandas as pd
import numpy as np
import sys
import os

#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
feature = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6
normalize = True

train_x_path = sys.argv[1]
train_y_path = sys.argv[2]
test_x_path = sys.argv[3]
output_path = sys.argv[4]


def acc(train_predict_result, train_label):
    t = 0
    for i in range(0, len(train_predict_result)):
        if train_predict_result[i][0] == train_label[i]:
            t += 1
    print(t/len(train_label))
    return


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


# train_data
np_train_label = train_label.values
np_data = train_data[feature].values
new_data = None
if normalize is True:
    all_mean = np.mean(np_data, axis=0)
    deviation = np.std(np_data, axis=0)
    new_data = (np_data-all_mean)/deviation
else:
    new_data = np_data


# test_data
np_test_data = test_data[feature].values
new_test_data = None
if normalize is True:
    new_test_data = (np_test_data-all_mean)/deviation
else:
    new_test_data = np_test_data


new_data_class0 = new_data[train_label['Y'] == 0]
new_data_class1 = new_data[train_label['Y'] == 1]
mean0 = np.mean(new_data_class0, axis=0)
mean1 = np.mean(new_data_class1, axis=0)
new_mean0 = mean0.reshape(len(mean0), 1)
new_mean1 = mean1.reshape(len(mean1), 1)


#get Σ∗

sigma0 = None
for i in range(0,len(new_data_class0)):
    if sigma0 is None:
        sigma0 = np.dot((new_data_class0[i].reshape(len(new_mean0),1)-new_mean0), (new_data_class0[i].reshape(len(new_mean0),1)-new_mean0).T)
    else:
        sigma0 += np.dot((new_data_class0[i].reshape(len(new_mean0),1)-new_mean0), (new_data_class0[i].reshape(len(new_mean0),1)-new_mean0).T)
sigma0 /= len(new_data_class0)

sigma1 = None
for i in range(0,len(new_data_class1)):
    if sigma1 is None:
        sigma1 = np.dot((new_data_class1[i].reshape(len(new_mean1),1)-new_mean1),(new_data_class1[i].reshape(len(new_mean1),1)-new_mean1).T)
    else:
        sigma1 += np.dot((new_data_class1[i].reshape(len(new_mean1),1)-new_mean1),(new_data_class1[i].reshape(len(new_mean1),1)-new_mean1).T)
sigma1 /= len(new_data_class1)
sigma = (len(new_data_class0)/(len(new_data_class0)+len(new_data_class1)))*sigma0 + (len(new_data_class1)/(len(new_data_class0)+len(new_data_class1)))*sigma1


wT = np.dot((new_mean0-new_mean1).T,np.linalg.inv(sigma))
b = -0.5*np.dot(np.dot(new_mean0.T,np.linalg.inv(sigma)),new_mean0)+0.5*np.dot(np.dot(new_mean1.T,np.linalg.inv(sigma)),new_mean1)+np.log(len(new_data_class0)/len(new_data_class1))


def predict(wT,b,x):
    return 1/(1+np.exp(-1*(np.dot(wT,x)+b)))


predict_result = np.zeros((len(new_test_data),1), dtype='int')-1
for i in range(0,len(new_test_data)):
    if predict(wT,b,new_test_data[i]) >= 0.5:
        predict_result[i][0] = 0
    else:
        predict_result[i][0] = 1
print(np.sum(predict_result))
print(len(predict_result))


train_predict_result = np.zeros((len(new_data),1), dtype='int')-1
for i in range(0,len(new_data)):
    if predict(wT,b,new_data[i]) >= 0.5:
        train_predict_result[i][0] = 0
    else:
        train_predict_result[i][0] = 1
print(np.sum(train_predict_result))
print(len(train_predict_result))
acc(train_predict_result, np_train_label)


# output public testing predict file
ans = pd.DataFrame()
ans = ans.assign(id=test_data_for_id)
ans = ans.assign(Value=predict_result)
print(ans)
ans.to_csv(output_path,index=False)
#ans = ans.assign(value=ans_value)