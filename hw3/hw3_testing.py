import sys
import os
import keras
import numpy as np
import pandas as pd

#arg
testing_data_path = sys.argv[1]
output_path = sys.argv[2]

#load testing data
test_data = pd.read_csv(testing_data_path, encoding='big5', dtype='O')
test_split_feature = test_data['feature'].str.split(' ')
test_feature = np.zeros((len(test_data),48,48,1), dtype='float32')

for index in range(0,len(test_split_feature)):
    c = np.asarray(test_split_feature[index], dtype=float)
    test_feature[index] = c.reshape(48,48,1)

model = keras.models.load_model('model15.h5')
prediction = model.predict_classes(test_feature)

# output public testing predict file
test_data_for_id = [str(i) for i in range(0,len(test_feature))]
ans = pd.DataFrame()
ans = ans.assign(id=test_data_for_id)
ans = ans.assign(label=prediction)
ans.to_csv(output_path,index=False)