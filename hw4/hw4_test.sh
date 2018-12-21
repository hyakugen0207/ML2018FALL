wget -O hw4_word.model.wv.vectors.npy https://www.dropbox.com/s/lxj2v0uli71hx3e/hw4_word.model.wv.vectors.npy?dl=1
wget -O hw4_word.model.trainables.syn1neg.npy https://www.dropbox.com/s/qfjugc5o5htsuya/hw4_word.model.trainables.syn1neg.npy?dl=1
wget -O hw4_word.model https://www.dropbox.com/s/8ovhpngzoafl7gt/hw4_word.model?dl=1
wget https://www.dropbox.com/s/jkghsqexzk6dbv8/model_1219_index0.h5?dl=1
wget https://www.dropbox.com/s/jv76fttze1k48rb/model_1219_index1.h5?dl=1
wget https://www.dropbox.com/s/j0989vcqav6co8f/model_1219_index2.h5?dl=1
wget https://www.dropbox.com/s/utbqjt7r08rdipx/model_1219_index3.h5?dl=1
wget https://www.dropbox.com/s/ydcesfx5z7pkhsr/model_1219_index4.h5?dl=1
wget https://www.dropbox.com/s/9rlsd1ea3kppa5t/model_1219_index5.h5?dl=1 

python3 hw4_test.py $1 $2 $3
