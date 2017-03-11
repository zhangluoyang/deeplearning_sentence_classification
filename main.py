# coding=utf-8
"""
深度学习应用于句子的情感分析
2017-03-11　张洛阳　南京
"""
from datasets import *
from model import *
from word2vec import word2vec
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="deeplearning for sentence classification")
    parser.add_argument('--model', help='neural network model, default is single_random_model', default='single_random_model', type=str)
    parser.add_argument('--embedding_size', help='word embedding size, default=200', default=200, type=int)
    parser.add_argument('--filter_sizes', help='all filter size for neural network, default is [3, 4, 5]', default=[3, 4, 5], type=list)
    parser.add_argument('--num_filters', help='the number of filters for convolution, default = 200', default=200, type=int)
    parser.add_argument('--dropout_keep_prob', help='dropout keep probability, default=0.5', default=0.5, type=float)
    parser.add_argument('--sequence_length', help='sentence\' length default=30', default=30, type=int)
    parser.add_argument('--num_classes', help='the number of labels, default=3', default=3, type=int)
    parser.add_argument('--batch_size', help='batch_size, default=64', default=64, type=int)
    parser.add_argument('--num_epochs', help='epochs, default=10', default=10, type=int)
    parser.add_argument('--train_data', help='train data', default='data/train.txt', type=str)
    parser.add_argument('--test_data', help='test data', default='data/test.txt', type=str)
    parser.add_argument('--word2vec', help='pretrain word2vec model', default='data/newsfinal.bin', type=str)
    parser.add_argument('--data_exists', help='datas has already processed, default=0', default=0, type=int)
    parser.add_argument('--train', help='train or test, default=train', default='train', type=str)
    return parser.parse_args()

def train_lstm_mean_model(args):
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data
    data_exists = args.data_exists
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vecMeanRnn(sequence_length=sequence_length, batch_size=batch_size, train_data=train_data,
                                   test_data=test_data, exists=data_exists, word2vec=w2c, embedding_size=embedding_size)
    params = {"embedding_size": embedding_size, "sequence_length": sequence_length, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "train_data": train_data, "test_data": test_data,
              "model": args.model}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    model = eval(args.model)(sequence_length, num_classes, embedding_size)
    model.fit(datas, num_epochs)

def test_lstm_mean_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    sequence_length = params['sequence_length']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_data = params['train_data']
    test_data = params['test_data']
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vecMeanRnnEval(sequence_length, batch_size, test_data, word2vec=w2c, embedding_size=embedding_size)
    model = eval(args.model)(sequence_length, num_classes, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(model.model_name), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model.load(checkpoint_file)
    model.eval(datas)

def train_bi_lstm_model(args):
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data
    data_exists = args.data_exists
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vecBiRnn(sequence_length=sequence_length, batch_size=batch_size, train_data=train_data,
                                   test_data=test_data, exists=data_exists, word2vec=w2c, embedding_size=embedding_size)
    params = {"embedding_size": embedding_size, "sequence_length": sequence_length, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "train_data": train_data, "test_data": test_data,
              "model": args.model}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    model = eval(args.model)(sequence_length, num_classes, embedding_size)
    model.fit(datas, num_epochs)

def test_bi_lstm_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    sequence_length = params['sequence_length']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_data = params['train_data']
    test_data = params['test_data']
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vecBiRnnEval(sequence_length, batch_size, test_data, word2vec=w2c, embedding_size=embedding_size)
    model = eval(args.model)(sequence_length, num_classes, embedding_size)
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(model.model_name), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model.load(checkpoint_file)
    model.eval(datas)

def train_single_random_model(args):
    embedding_size = args.embedding_size
    filter_sizes = args.filter_sizes
    num_filters = args.num_filters
    dropout_keep_prob = args.dropout_keep_prob
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data
    data_exists = args.data_exists
    params = {"embedding_size": embedding_size, "filter_sizes": filter_sizes, "num_filters": num_filters,
              "dropout_keep_prob": dropout_keep_prob, "sequence_length": sequence_length, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "train_data": train_data, "test_data": test_data,
              "model": args.model}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    datas = DataSetWordIndex(sequence_length, batch_size, train_data, test_data, data_exists)
    vocab_size = datas.vocab_size
    model = eval(args.model)(sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters)
    model.fit(datas, num_epochs, keep_prob=dropout_keep_prob)

def test_single_random_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    filter_sizes = params['filter_sizes']
    num_filters = params['num_filters']
    dropout_keep_prob = params['dropout_keep_prob']
    sequence_length = params['sequence_length']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_data = params['train_data']
    test_data = params['test_data']
    datas = DataSetWordIndexEval(sequence_length, batch_size, test_data)
    vocab_size = datas.vocab_size
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters)
    model.load(checkpoint_file)
    model.eval(datas)

def train_single_static_model(args):
    filter_sizes = args.filter_sizes
    num_filters = args.num_filters
    dropout_keep_prob = args.dropout_keep_prob
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data
    data_exists = args.data_exists
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vec(sequence_length=sequence_length, batch_size=batch_size, train_data=train_data,
                            test_data=test_data, exists=data_exists, word2vec=w2c, embedding_size=embedding_size)
    params = {"embedding_size": embedding_size, "filter_sizes": filter_sizes, "num_filters": num_filters,
              "dropout_keep_prob": dropout_keep_prob, "sequence_length": sequence_length, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "train_data": train_data, "test_data": test_data,
              "model": args.model}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    vocab_size = datas.vocab_size
    model = eval(args.model)(sequence_length, num_classes, embedding_size, filter_sizes, num_filters)
    model.fit(datas, num_epochs, keep_prob=dropout_keep_prob)

def test_single_static_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    filter_sizes = params['filter_sizes']
    num_filters = params['num_filters']
    dropout_keep_prob = params['dropout_keep_prob']
    sequence_length = params['sequence_length']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_data = params['train_data']
    test_data = params['test_data']

    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    datas = DataSetWord2vecEval(sequence_length=sequence_length, batch_size=batch_size, train_data=train_data,
                                test_data=test_data, word2vec=w2c, embedding_size=embedding_size)
    vocab_size = datas.vocab_size
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = eval(args.model)(sequence_length, num_classes, embedding_size, filter_sizes, num_filters)
    model.load(checkpoint_file)
    model.eval(datas)

def train_single_truned_model(args):
    # embedding_size = args.embedding_size
    filter_sizes = args.filter_sizes
    num_filters = args.num_filters
    dropout_keep_prob = args.dropout_keep_prob
    sequence_length = args.sequence_length
    num_classes = args.num_classes
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    train_data = args.train_data
    test_data = args.test_data
    data_exists = args.data_exists
    w2c = word2vec(args.word2vec).word2vec
    embedding_size = word2vec(args.word2vec).embedding_size
    params = {"embedding_size": embedding_size, "filter_sizes": filter_sizes, "num_filters": num_filters,
              "dropout_keep_prob": dropout_keep_prob, "sequence_length": sequence_length, "num_classes": num_classes,
              "batch_size": batch_size, "num_epochs": num_epochs, "train_data": train_data, "test_data": test_data,
              "model": args.model}
    if not os.path.exists("conf"):
        os.mkdir("conf")
    json.dump(params, open("conf/{}.json".format(args.model), "w"))
    datas = DataSetWord2vecTruned(sequence_length, batch_size, train_data, test_data, exists=data_exists, word2vec=w2c,
                                  embedding_size=embedding_size)
    word2vec_init = datas.word_vec
    model = eval(args.model)(sequence_length, num_classes, embedding_size, filter_sizes, num_filters, word2vec_init)
    model.fit(datas, num_epochs, dropout_keep_prob)

def test_single_truned_model(args):
    params = json.load(open("conf/{}.json".format(args.model), "r"))
    embedding_size = params['embedding_size']
    filter_sizes = params['filter_sizes']
    num_filters = params['num_filters']
    dropout_keep_prob = params['dropout_keep_prob']
    sequence_length = params['sequence_length']
    num_classes = params['num_classes']
    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    train_data = params['train_data']
    test_data = params['test_data']
    datas = DataSetWord2vecTrunedEval(sequence_length, batch_size, test_data)
    vocab_size = datas.vocab_size
    checkpoint_dir = os.path.abspath(os.path.join("{}".format(args.model), "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    # 这里的矩阵初始化部分可以不去考虑
    model = eval(args.model)(sequence_length, num_classes, embedding_size, filter_sizes, num_filters, word2vec_init=np.zeros((vocab_size, embedding_size), dtype="float32"))
    model.load(checkpoint_file)
    model.eval(datas)

def main(args):
    model_name = args.model
    train_or_test = args.train
    if train_or_test == "train":
        if model_name == "lstm_mean_model":
            train_lstm_mean_model(args)
        elif model_name == "bi_lstm_model":
            train_bi_lstm_model(args)
        elif model_name == "single_random_model":
            train_single_random_model(args)
        elif model_name == "single_static_model":
            train_single_static_model(args)
        elif model_name == "single_truned_model":
            train_single_truned_model(args)
    elif train_or_test == "test":
        if model_name == "lstm_mean_model":
            test_lstm_mean_model(args)
        elif model_name == "bi_lstm_model":
            test_bi_lstm_model(args)
        elif model_name == "single_random_model":
            test_single_random_model(args)
        elif model_name == "single_static_model":
            test_single_static_model(args)
        elif model_name == "single_truned_model":
            test_single_truned_model(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)


