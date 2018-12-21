#--------------------------------
# import
#--------------------------------
import numpy as np
import pandas as pd
import sys
import os
import jieba
import logging
import gensim
import keras
import random
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, Embedding, Bidirectional, GRU, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

#--------------------------------
#--------------------------------
# all parameter we need
#--------------------------------
#--------------------------------
local = False
word_len = [64,64,64,64,64,64]
vector_size = [256,256,256,256,256,256]
seed = 87451
epoch = [30,20,30,20,20,30]
batch = [96,64,72,128,96,72]
early = 20
drop = [0.5,0.6,0.4,0.5,0.5,0.45]
drop_r = [0.5,0.6,0.4,0.5,0.5,0.45]
L = [200,256,168,227,150,250]
D = [76,64,84,96,84,64]
d2 = [0.5,0.6,0.4,0.5,0.6,0.5]

def encode(data):
    print('Shape of data (BEFORE encode): %s' % str(data.shape))
    encoded = to_categorical(data)
    print('Shape of data (AFTER  encode): %s\n' % str(encoded.shape))
    return encoded

def build_LSTM_model(index):
    if index==0:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index]))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif index==1:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index])))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(D[vali_index]+50))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif index==2:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index]))
        model.add(Dense(D[vali_index]+30))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif index==3:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index])))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    elif index==4:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(Bidirectional(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index])))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    else:
        model = Sequential()
        model.add(Embedding(len(word2idx),vector_size[vali_index],weights=[embeddings_matrix],input_length=word_len[vali_index],trainable=False))
        model.add(BatchNormalization())
        model.add(LSTM(L[vali_index], implementation=2, dropout=drop[vali_index], recurrent_dropout=drop_r[vali_index]))
        model.add(Dense(D[vali_index]))
        model.add(BatchNormalization())
        model.add(Dropout(d2[vali_index]))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


for vali_index in range(0, 6):
    version_name = '1219_index' + str(vali_index)
    model_name = 'model_' + version_name + '.h5'
    file_name = 'result_' + version_name + '.csv'
    stopWords = []

    train_data = np.zeros((120000, word_len[vali_index], vector_size[vali_index]), dtype=float)
    train_label = np.zeros((120000,), dtype=int)
    test_data = np.zeros((80000, word_len[vali_index], vector_size[vali_index]), dtype=float)
    # file path
    train_path = None
    train_label_path = None
    test_path = None
    output_path = None
    dic_path = None
    stopword_path = 'StopWord'
    if local is True:
        train_path = 'train_x.csv'
        test_path = 'test_x.csv'
        train_label_path = 'train_y.csv'
        dic_path = 'dict.txt.big'
        output_path = 'ans.csv'
    else:
        # test  bash hw4_test.sh <test_x file> <dict.txt.big file> <output file>
        # train bash hw4_train.sh <train_x file> <train_y file> <dict.txt.big file>
        train_path = sys.argv[1]
        train_label_path = sys.argv[2]
        dic_path = sys.argv[4]
        output_path = 'ans.csv'
        test_path = sys.argv[3]

    # --------------------------------
    # load stopwords
    # --------------------------------
    with open(stopword_path, 'r', encoding='UTF-8') as file:
        for data in file.readlines():
            data = data.strip()
            stopWords.append(data)

    # --------------------------------
    # load training data
    # --------------------------------
    file = open(train_path, 'r')
    read_text = file.readline()
    read_text = file.readline()
    train_data_list = []
    while read_text != '':
        index = 0
        while read_text[index] != ',':
            index += 1
        train_data_list.append(read_text[index + 1:len(read_text)].splitlines()[0])
        read_text = file.readline()

    # --------------------------------
    # load testing data
    # --------------------------------
    file = open(test_path, 'r')
    read_text = file.readline()
    read_text = file.readline()
    test_data_list = []
    while read_text != '':
        index = 0
        while read_text[index] != ',':
            index += 1
        test_data_list.append(read_text[index + 1:len(read_text)].splitlines()[0])
        read_text = file.readline()

    # --------------------------------
    # load training label
    # --------------------------------
    label = pd.read_csv(train_label_path)
    train_label = label['label']

    # --------------------------------
    # 切句子 + 補全
    # --------------------------------
    jieba.set_dictionary(dic_path)
    train_cut = []
    for index in range(len(train_data_list)):
        seg_list = jieba.cut(train_data_list[index], cut_all=False)
        train_cut.append(list(filter(lambda a: a not in stopWords and a != '\n', seg_list)))
        if index % 1000 == 0:
            print(index)

    test_cut = []
    for index in range(len(test_data_list)):
        seg_list = jieba.cut(test_data_list[index], cut_all=False)
        test_cut.append(list(filter(lambda a: a not in stopWords and a != '\n', seg_list)))
        if index % 1000 == 0:
            print(index)

    # --------------------------------
    # make word_embedding model
    # --------------------------------
    word_embedding_data = train_cut + test_cut
    word_model = gensim.models.Word2Vec(word_embedding_data, min_count=1, size=vector_size[vali_index])
    # 保存模型，供日後使用
    word_model.save("word2vec2.model")

    testtt = train_cut.copy()
    shuffle_index = [i for i in range(len(testtt))]
    random.Random(seed).shuffle(shuffle_index)

    word2idx = {"_PAD": 0}
    vocab_list = [(k, word_model.wv[k]) for k, v in word_model.wv.vocab.items()]
    # 存储所有 word2vec 中所有向量的数组，留意其中多一位，词向量全为 0， 用于 padding
    embeddings_matrix = np.zeros((len(word_model.wv.vocab.items()) + 1, word_model.vector_size))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]

    train_data_encoded = np.zeros((100000, word_len[vali_index]))
    val_data_encoded = np.zeros((20000, word_len[vali_index]))
    one_hot_label = encode(train_label)
    train_label_new = np.zeros((100000, 2))
    val_label = np.zeros((20000, 2))
    count_index = 0
    for index in range(len(train_cut)):
        encoded_docs = np.zeros((word_len[vali_index],))
        for i in range(min(len(train_cut[shuffle_index[index]]), word_len[vali_index])):
            encoded_docs[i] = word2idx[train_cut[shuffle_index[index]][i]]
        if index >= vali_index * 20000 and index < (vali_index + 1) * 20000:
            val_label[index - vali_index * 20000] = one_hot_label[shuffle_index[index]]
            val_data_encoded[index - vali_index * 20000] = encoded_docs
        else:
            train_label_new[count_index] = one_hot_label[shuffle_index[index]]
            train_data_encoded[count_index] = encoded_docs
            count_index += 1

    early_stopping = EarlyStopping(monitor='val_acc', patience=early, verbose=1, mode='max')
    saveBestModel = ModelCheckpoint(model_name, monitor="val_acc", save_best_only=True, save_weights_only=False)

    model = build_LSTM_model(vali_index)

    model.fit(train_data_encoded, train_label_new, epochs=epoch[vali_index], batch_size=batch[vali_index],
              validation_data=(val_data_encoded, val_label), callbacks=[early_stopping, saveBestModel])

test_data_encode = np.zeros((len(test_cut), word_len[vali_index]))
for index in range(len(test_cut)):
    encoded_docs = np.zeros((word_len[vali_index],))
    for i in range(min(len(test_cut[index]), word_len[vali_index])):
        encoded_docs[i] = word2idx[test_cut[index][i]]
    test_data_encode[index] = encoded_docs

m = []
model_name = ['model_1219_index0.h5', 'model_1219_index1.h5', 'model_1219_index2.h5', 'model_1219_index3.h5',
              'model_1219_index4.h5', 'model_1219_index5.h5']
for i in range(len(model_name)):
    m.append(keras.models.load_model(model_name[i]))

m_predict = []
test_data_for_id = [str(i) for i in range(0, len(test_data_encode))]
for i in range(len(model_name)):
    m_ans = m[i].predict(test_data_encode)
    print(m_ans)
    m_predict.append(m_ans)

##平均
ccc = 0
b = np.zeros((len(m_predict[0]),), dtype=int)
for i in range(len(m_predict[0])):
    num = [0.0, 0.0]
    for j in range(len(model_name)):
        num += m_predict[j][i]
    aveg = num / len(model_name)
    if aveg[1] >= 0.5:
        b[i] = 1
    else:
        b[i] = 0

# output public testing predict file 平均
test_data_for_id = [str(i) for i in range(0, len(test_data_encode))]
m_ans2 = pd.DataFrame()
m_ans2 = m_ans2.assign(id=test_data_for_id)
m_ans2 = m_ans2.assign(label=b)
m_ans2.to_csv('avg_' + file_name, index=False)