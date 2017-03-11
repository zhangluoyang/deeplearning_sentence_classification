deeplearning for sentence classification
author:zhangluoyang email:55058629@qq.com
environment dependence
python2.7
tensorflow0.12
jieba
cPick
numpy
datasets: train.txt test.txt
format example:
    label   sentence
    1	晋西车轴(600495)拟使用2亿元购买信托产品
word2vec pretrain bin file
    vector.bin
word2vec.py: a util for word2vec
model.py: neural network model
datasets: normal data format for neural network
I have already realized 5 methods, please see the code for detail infomation.
how to run:
    python main.py --model single_random_model --train train --num_epoch 20
    python main.py --model single_random_model --train test

    python main.py --model single_static_model --train train --num_epoch 20
    python main.py --model single_static_model --train test

    python main.py --model single_truned_model --train train --num_epoch 20
    python main.py --model single_truned_model --train test

    python main.py --model lstm_mean_model --train train --num_epoch 20
    python main.py --model lstm_mean_model --train test

    python main.py --model bi_lstm_model --train train --num_epoch 20
    python main.py --model bi_lstm_model --train test
reference:
    1 Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.
    2 LSTM Networks for Sentiment Analysis http://deeplearning.net/tutorial/lstm.html
    3 Word2vecSentimentRNN https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment





