import pandas as pd
import numpy as np


# In[2]:


#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#feature = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
feature = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','PAY_0_no_use','PAY_2_no_use','PAY_3_no_use','PAY_4_no_use','PAY_5_no_use','PAY_6_no_use','PAY_0_payoff','PAY_2_payoff','PAY_3_payoff','PAY_4_payoff','PAY_5_payoff','PAY_6_payoff']
#LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6
normalize = True


# In[3]:


train_data = pd.read_csv('train_x.csv', encoding='big5', dtype='float')
train_label = pd.read_csv('train_y.csv', encoding='big5', dtype='float64')
test_data = pd.read_csv('test_x.csv', encoding='big5', dtype='float64')
test_data_for_id = ['id_'+str(i) for i in range(0,10000)]
# modify feature of PAY_X
for var in train_data.columns[5:11]:
    train_data[var + '_no_use'] = np.where(train_data[var] == -2, 1, 0)
    train_data[var + '_payoff'] = np.where(train_data[var] == -1, 1, 0)
    train_data[var] = np.where(train_data[var] < 0, 0, train_data[var])
    test_data[var + '_no_use'] = np.where(test_data[var] == -2, 1, 0)
    test_data[var + '_payoff'] = np.where(test_data[var] == -1, 1, 0)
    test_data[var] = np.where(test_data[var] < 0, 0, test_data[var])
#train_data


# In[4]:


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

print(new_train_data)


# In[6]:


def loss(w, bias, data, label):
    cross_entropy = 0
    for i in range(0,len(data)):
        f = 1/(1+np.exp(-1*(np.dot(data[i],w)+bias)))
        if f == 0:
            f = 0.00000001
        elif f == 1:
            f = 0.99999999
        cross_entropy += -1*(label[i]*np.log(f)+(1-label[i])*np.log(1-f))
    return cross_entropy


# In[7]:


def acc(train_predict_result, train_label):
    t = 0
    for i in range(0, len(train_predict_result)):
        if train_predict_result[i][0] == train_label[i]:
            t += 1
    print(t/len(train_data))


# In[8]:


def gradient(w, bias, data, label, lr, for_w):
    if for_w is True:
        new_w = w.copy()
        for i in range(0,len(w)):
            result = 0
            for j in range(0,len(data)):
                result -= (label[j]-1/(1+np.exp(-1*(np.dot(data[j],w)+bias))))*data[j][i]
            new_w[i] = w[i] - lr*result
        return new_w
    else:
        result = 0
        for j in range(0,len(data)):
            result -= (label[j]-1/(1+np.exp(-1*(np.dot(data[j],w)+bias))))
        return bias - lr*result


# In[9]:



lr = 0.0001
batch = 50
interaction = 10000
pre_batch = 0
w = np.zeros((len(feature),1))+1
bias = 4.423

for i in range(0,interaction):
    batch_data = None
    batch_label = None
    if pre_batch + batch >= len(train_data):
        batch_data = new_train_data[pre_batch:len(train_data)]
        batch_label = np_label_data[pre_batch:len(train_data)]
        pre_batch = 0
    else:
        batch_data = new_train_data[pre_batch:pre_batch+batch]
        batch_label = np_label_data[pre_batch:pre_batch+batch]
        pre_batch += batch
    new_w = gradient(w, bias, batch_data, batch_label, lr, True)
    new_bias = gradient(w, bias, batch_data, batch_label, lr, False)
    w = new_w
    bias = new_bias
    if i%100 == 0:
        print(loss(w, bias, new_train_data, np_label_data),i)


# In[ ]:





# In[ ]:





# In[10]:


def predict(w,b,x):
    return 1/(1+np.exp(-1*(np.dot(x,w)+b)))

train_predict_result = np.zeros((len(new_train_data),1), dtype='int')-1
for i in range(0,len(new_train_data)):
    if predict(w,bias,new_train_data[i]) >= 0.5:
        train_predict_result[i][0] = 1
    else:
        train_predict_result[i][0] = 0
print(np.sum(train_predict_result))
print(len(train_predict_result))
acc(train_predict_result,np_label_data)


# In[11]:


def acc(train_predict_result, train_label):
    t = 0
    for i in range(0, len(train_predict_result)):
        if train_predict_result[i][0] == train_label[i]:
            t += 1
    print(t/len(train_data))


# In[12]:



predict_result = np.zeros((len(new_test_data),1), dtype='int')-1
for i in range(0,len(new_test_data)):
    if predict(w,bias,new_test_data[i]) >= 0.5:
        predict_result[i][0] = 1
    else:
        predict_result[i][0] = 0
print(np.sum(predict_result))
print(len(predict_result))


# In[13]:


# output public testing predict file
ans = pd.DataFrame()
ans = ans.assign(id=test_data_for_id)
ans = ans.assign(Value=predict_result)
print(ans)
ans.to_csv('homework2_4_logistic_normalize.csv',index=False)
#ans = ans.assign(value=ans_value)


# In[ ]:





# In[ ]:




