# coding=utf-8
"""
深度学习应用于句子的情感分析
2017-03-11　张洛阳　南京
"""
import jieba
import numpy as np
import cPickle
import os
class DataSetWord2vecMeanRnn(object):

    def __init__(self, sequence_length, batch_size, train_data, test_data, exists=False, word2vec=None, embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        if exists:
            print("loading....")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/lstm_mean/all_words", "r"))
        all_labels = cPickle.load(open("data/lstm_mean/all_labels", "r"))
        instances = cPickle.load(open("data/lstm_mean/instances", "r"))
        words_id = cPickle.load(open("data/lstm_mean/words_id", "r"))
        labels_id = cPickle.load(open("data/lstm_mean/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def process(self):
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        all_words = set()
        all_labels = set()
        for line in lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            map(lambda word: all_words.add(word), words)
            all_labels.add(label)
        instances = []
        for line in train_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)

        all_words.add('unknow')
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        if not os.path.exists("data/lstm_mean"):
            os.mkdir("data/lstm_mean")
        cPickle.dump(all_words, open("data/lstm_mean/all_words", "w"))
        cPickle.dump(all_labels, open("data/lstm_mean/all_labels", "w"))
        cPickle.dump(instances, open("data/lstm_mean/instances", "w"))
        cPickle.dump(words_id, open("data/lstm_mean/words_id", "w"))
        cPickle.dump(labels_id, open("data/lstm_mean/labels_id", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecMeanRnnEval(object):
    def __init__(self, sequence_length, batch_size, test_data, word2vec=None,
                 embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/lstm_mean/all_words", "r"))
        all_labels = cPickle.load(open("data/lstm_mean/all_labels", "r"))
        words_id = cPickle.load(open("data/lstm_mean/words_id", "r"))
        labels_id = cPickle.load(open("data/lstm_mean/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
        instances = []
        for line in test_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecMeanRnnConvert(object):
    def __init__(self, sequence_length, word2vec=None):
        """
        :param sequence_length: 输入句子的长度
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.word2vec = word2vec
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/lstm_mean/all_words", "r"))
        all_labels = cPickle.load(open("data/lstm_mean/all_labels", "r"))
        words_id = cPickle.load(open("data/lstm_mean/words_id", "r"))
        labels_id = cPickle.load(open("data/lstm_mean/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id


    def convert(self, sentence):
        """

        :param sentence: 未分词的单个句子
        :return:
        """
        words = jieba.cut(sentence, cut_all=False)
        words = map(lambda word: word.encode('utf-8'), words)
        if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
            words = words[0:self.sequence_length]
        else:  # 长度不足进行填充
            words = (self.sequence_length - len(words)) * ['unknow'] + words
        vec_data = np.zeros(shape=(1, self.sequence_length, self.embedding_size))
        for j in range(len(words)):
            word = words[j]
            if word in self.word2vec:
                v = self.word2vec[word].reshape(1, self.embedding_size)
            else:
                v = np.zeros(shape=(1, self.embedding_size))
            vec_data[0][j] = v
        return vec_data


class DataSetWord2vecBiRnn(object):

    def __init__(self, sequence_length, batch_size, train_data, test_data, exists=False, word2vec=None, embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        if exists:
            print("loading....")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/bi_lstm/all_words", "r"))
        all_labels = cPickle.load(open("data/bi_lstm/all_labels", "r"))
        instances = cPickle.load(open("data/bi_lstm/instances", "r"))
        words_id = cPickle.load(open("data/bi_lstm/words_id", "r"))
        labels_id = cPickle.load(open("data/bi_lstm/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def process(self):
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        all_words = set()
        all_labels = set()
        for line in lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            map(lambda word: all_words.add(word), words)
            all_labels.add(label)
        instances = []
        for line in train_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)

        all_words.add('unknow')
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        if not os.path.exists("data/bi_lstm"):
            os.mkdir("data/bi_lstm")
        cPickle.dump(all_words, open("data/bi_lstm/all_words", "w"))
        cPickle.dump(all_labels, open("data/bi_lstm/all_labels", "w"))
        cPickle.dump(instances, open("data/bi_lstm/instances", "w"))
        cPickle.dump(words_id, open("data/bi_lstm/words_id", "w"))
        cPickle.dump(labels_id, open("data/bi_lstm/labels_id", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecBiRnnEval(object):
    def __init__(self, sequence_length, batch_size, test_data, word2vec=None,
                 embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/bi_lstm/all_words", "r"))
        all_labels = cPickle.load(open("data/bi_lstm/all_labels", "r"))
        words_id = cPickle.load(open("data/bi_lstm/words_id", "r"))
        labels_id = cPickle.load(open("data/bi_lstm/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
        instances = []
        for line in test_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecBiRnnConvert(object):
    def __init__(self, sequence_length, word2vec=None):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.word2vec = word2vec
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/bi_lstm/all_words", "r"))
        all_labels = cPickle.load(open("data/bi_lstm/all_labels", "r"))
        words_id = cPickle.load(open("data/bi_lstm/words_id", "r"))
        labels_id = cPickle.load(open("data/bi_lstm/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id


    def convert(self, sentence):
        """

        :param sentence: 未分词的单个句子
        :return:
        """
        words = jieba.cut(sentence, cut_all=False)
        words = map(lambda word: word.encode('utf-8'), words)
        if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
            words = words[0:self.sequence_length]
        else:  # 长度不足进行填充
            words = (self.sequence_length - len(words)) * ['unknow'] + words
        vec_data = np.zeros(shape=(1, self.sequence_length, self.embedding_size))
        for j in range(len(words)):
            word = words[j]
            if word in self.word2vec:
                v = self.word2vec[word].reshape(1, self.embedding_size)
            else:
                v = np.zeros(shape=(1, self.embedding_size))
            vec_data[0][j] = v
        return vec_data


class DataSetWordIndex(object):
    def __init__(self, sequence_length, batch_size, train_data, test_data, exists=False):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        if exists:
            print("loading....")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_random_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_random_cnn/all_labels", "r"))
        instances = cPickle.load(open("data/single_random_cnn/instances", "r"))
        words_id = cPickle.load(open("data/single_random_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_random_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def process(self):
        import jieba
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        all_words = set()
        all_labels = set()
        for line in lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            map(lambda word: all_words.add(word), words)
            all_labels.add(label)
        instances = []
        for line in train_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)

        all_words.add('unknow')
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        if not os.path.exists("data/single_random_cnn"):
            os.mkdir("data/single_random_cnn")
        cPickle.dump(all_words, open("data/single_random_cnn/all_words", "w"))
        cPickle.dump(all_labels, open("data/single_random_cnn/all_labels", "w"))
        cPickle.dump(instances, open("data/single_random_cnn/instances", "w"))
        cPickle.dump(words_id, open("data/single_random_cnn/words_id", "w"))
        cPickle.dump(labels_id, open("data/single_random_cnn/labels_id", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        # word -> id
        batch_instances_word_ids = []
        batch_instances_labels = []
        for instance in batch_instances:
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words))*['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'], words)
            batch_instances_word_ids.append(word_ids)
            batch_instances_labels.append(label)
        return batch_instances_word_ids, batch_instances_labels


class DataSetWordIndexEval(object):
    def __init__(self, sequence_length, batch_size, test_data):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_data = test_data
        self.load()  # 加载训练数据信息
        self.generate_data()  # 生成数据信息
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_random_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_random_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_random_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_random_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        import jieba
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
        instances = []
        for line in test_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        self.instances = instances
        self.num_examples = len(self.instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        # word -> id
        batch_instances_word_ids = []
        batch_instances_labels = []
        for instance in batch_instances:
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words))*['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'], words)
            batch_instances_word_ids.append(word_ids)
            batch_instances_labels.append(label)
        return batch_instances_word_ids, batch_instances_labels


class DataSetWordIndexConvert(object):
    def __init__(self, sequence_length):
        """
        :param sequence_length: 输入句子的长度
        """
        self.sequence_length = sequence_length
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/single_random_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_random_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_random_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_random_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def convert(self, sentence):
        """

        :param sentence: 未分词的单个句子
        :return:
        """
        words = jieba.cut(sentence, cut_all=False)
        words = map(lambda word: word.encode('utf-8'), words)
        if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
            words = words[0:self.sequence_length]
        else:  # 长度不足进行填充
            words = (self.sequence_length - len(words)) * ['unknow'] + words
        word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'], words)
        return [word_ids]


class DataSetWord2vec(object):

    def __init__(self, sequence_length, batch_size, train_data, test_data, exists=False, word2vec=None, embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        if exists:
            print("loading....")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_static_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_static_cnn/all_labels", "r"))
        instances = cPickle.load(open("data/single_static_cnn/instances", "r"))
        words_id = cPickle.load(open("data/single_static_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_static_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def process(self):
        import jieba
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        all_words = set()
        all_labels = set()
        for line in lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            map(lambda word: all_words.add(word), words)
            all_labels.add(label)
        instances = []
        for line in train_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)

        all_words.add('unknow')
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))
        if not os.path.exists("data/single_static_cnn"):
            os.mkdir("data/single_static_cnn")
        cPickle.dump(all_words, open("data/single_static_cnn/all_words", "w"))
        cPickle.dump(all_labels, open("data/single_static_cnn/all_labels", "w"))
        cPickle.dump(instances, open("data/single_static_cnn/instances", "w"))
        cPickle.dump(words_id, open("data/single_static_cnn/words_id", "w"))
        cPickle.dump(labels_id, open("data/single_static_cnn/labels_id", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecEval(object):
    def __init__(self, sequence_length, batch_size, train_data, test_data, word2vec=None,
                 embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        self.load()
        self.generate_data()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_static_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_static_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_static_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_static_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        import jieba
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
        instances = []
        for line in test_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        self.instances = instances
        self.num_examples = len(self.instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        vec_batch_datas = np.zeros(shape=(len(batch_instances), self.sequence_length, self.embedding_size))
        # word -> vec
        batch_instances_labels = []
        for i in range(len(batch_instances)):
            instance = batch_instances[i]
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            for j in range(len(words)):
                word = words[j]
                if word in self.word2vec:
                    v = self.word2vec[word].reshape(1, self.embedding_size)
                else:
                    v = np.zeros(shape=(1, self.embedding_size))
                vec_batch_datas[i][j] = v
            batch_instances_labels.append(label)
        return vec_batch_datas, batch_instances_labels


class DataSetWord2vecConvert(object):

    def __init__(self, sequence_length, word2vec=None):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        :param word2vec: 预先处理的word2vec dict类型
        :param embedding_size: word2vec当中的embedding_size
        """
        self.sequence_length = sequence_length
        self.word2vec = word2vec
        self.load()

    def load(self):
        all_words = cPickle.load(open("data/single_static_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_static_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_static_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_static_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def convert(self, sentence):
        """
        :param sentence: 未分词的单个句子
        :return:
        """
        words = jieba.cut(sentence, cut_all=False)
        words = map(lambda word: word.encode('utf-8'), words)
        if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
            words = words[0:self.sequence_length]
        else:  # 长度不足进行填充
            words = (self.sequence_length - len(words)) * ['unknow'] + words
        vec_data = np.zeros(shape=(1, self.sequence_length, self.embedding_size))
        for j in range(len(words)):
            word = words[j]
            if word in self.word2vec:
                v = self.word2vec[word].reshape(1, self.embedding_size)
            else:
                v = np.zeros(shape=(1, self.embedding_size))
            vec_data[0][j] = v
        return vec_data


class DataSetWord2vecTruned(object):
    def __init__(self, sequence_length, batch_size, train_data, test_data, exists=False, word2vec=None, embedding_size=200):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param train_data: 训练数据集
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_data = train_data
        self.test_data = test_data
        self.word2vec = word2vec
        self.embedding_size = embedding_size
        if exists:
            print("loading....")
            self.load()
        else:
            print("processing....")
            self.process()
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_truned_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_truned_cnn/all_labels", "r"))
        instances = cPickle.load(open("data/single_truned_cnn/instances", "r"))
        words_id = cPickle.load(open("data/single_truned_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_truned_cnn/labels_id", "r"))
        word_vec = cPickle.load(open("data/single_truned_cnn/word_vec", "r"))
        self.word_vec = word_vec
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def process(self):
        import jieba
        with open(self.train_data, "r") as f:
            train_lines = f.readlines()
            lines = train_lines
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
            lines = lines + test_lines
        all_words = set()
        all_labels = set()
        for line in lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            map(lambda word: all_words.add(word), words)
            all_labels.add(label)
        instances = []
        for line in train_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        all_words.add('unknow')
        all_words = list(all_words)
        all_labels = list(all_labels)
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        words_id = dict(zip(all_words, range(vocab_size)))
        labels_id = dict(zip(all_labels, range(labels_size)))

        # 生成词向量表
        word_vec = np.zeros(shape=(len(all_words), self.embedding_size), dtype="float32")
        for i in range(len(all_words)):
            word = all_words[i]
            if word in self.word2vec:
                word_vec[i] = self.word2vec[word]
            else:
                word_vec[i] = np.random.rand(1, self.embedding_size)
        self.word_vec = word_vec

        if not os.path.exists("data/single_truned_cnn"):
            os.mkdir("data/single_truned_cnn")
        cPickle.dump(all_words, open("data/single_truned_cnn/all_words", "w"))
        cPickle.dump(all_labels, open("data/single_truned_cnn/all_labels", "w"))
        cPickle.dump(instances, open("data/single_truned_cnn/instances", "w"))
        cPickle.dump(words_id, open("data/single_truned_cnn/words_id", "w"))
        cPickle.dump(labels_id, open("data/single_truned_cnn/labels_id", "w"))
        cPickle.dump(word_vec, open("data/single_truned_cnn/word_vec", "w"))
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id
        self.instances = instances
        self.num_examples = len(instances)

    def shuffle(self):
        np.random.shuffle(self.index)
        self.point = 0

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        # word -> id
        batch_instances_word_ids = []
        batch_instances_labels = []
        for instance in batch_instances:
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'],
                           words)
            batch_instances_word_ids.append(word_ids)
            batch_instances_labels.append(label)
        return batch_instances_word_ids, batch_instances_labels


class DataSetWord2vecTrunedEval(object):
    def __init__(self, sequence_length, batch_size, test_data):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        """
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.test_data = test_data
        self.load()  # 加载训练数据信息
        self.generate_data()  # 生成数据信息
        self.batch_nums = self.num_examples // self.batch_size
        self.index = np.arange(self.num_examples)
        self.point = 0

    def load(self):
        all_words = cPickle.load(open("data/single_truned_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_truned_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_truned_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_truned_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def generate_data(self):
        import jieba
        with open(self.test_data, "r") as f:
            test_lines = f.readlines()
        instances = []
        for line in test_lines:
            label_sentence = line.split('\t')
            label = label_sentence[0]
            sentence = label_sentence[1]
            words = jieba.cut(sentence, cut_all=False)
            words = map(lambda word: word.encode('utf-8'), words)
            if len(words) > self.sequence_length: continue  # 如果句子长度过长 暂时去除
            instance = ([words, label])
            instances.append(instance)
        self.instances = instances
        self.num_examples = len(self.instances)

    def next_batch(self):
        start = self.point
        self.point = self.point + self.batch_size
        if self.point > self.num_examples:
            self.shuffle()
            start = 0
            self.point = self.point + self.batch_size
        end = self.point
        batch_instances = map(lambda x: self.instances[x], self.index[start:end])
        # word -> id
        batch_instances_word_ids = []
        batch_instances_labels = []
        for instance in batch_instances:
            words = instance[0]
            if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
                words = words[0:self.sequence_length]
            else:  # 长度不足进行填充
                words = (self.sequence_length - len(words)) * ['unknow'] + words
            label = instance[1]
            label_id = self.labels_id[label]
            label = [0] * self.labels_size  # one-hot 类型编码
            label[label_id] = 1
            word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'],
                           words)
            batch_instances_word_ids.append(word_ids)
            batch_instances_labels.append(label)
        return batch_instances_word_ids, batch_instances_labels


class DataSetWord2vecTrunedConvert(object):
    def __init__(self, sequence_length):
        """
        :param sequence_length: 输入句子的长度
        :param batch_size:
        :param test_data: 测试数据集
        :param exists: 数据源是否已经加载完成
        """
        self.sequence_length = sequence_length
        self.load()  # 加载训练数据信息

    def load(self):
        all_words = cPickle.load(open("data/single_truned_cnn/all_words", "r"))
        all_labels = cPickle.load(open("data/single_truned_cnn/all_labels", "r"))
        words_id = cPickle.load(open("data/single_truned_cnn/words_id", "r"))
        labels_id = cPickle.load(open("data/single_truned_cnn/labels_id", "r"))
        vocab_size = len(all_words)
        labels_size = len(all_labels)
        self.labels_id = labels_id
        self.all_words = all_words
        self.all_labels = all_labels
        self.vocab_size = vocab_size
        self.labels_size = labels_size
        self.words_id = words_id

    def convert(self, sentence):
        """

        :param sentence: 未分词的单个句子
        :return:
        """
        words = jieba.cut(sentence, cut_all=False)
        words = map(lambda word: word.encode('utf-8'), words)
        if len(words) > self.sequence_length:  # 超出长度范围去除句子尾部
            words = words[0:self.sequence_length]
        else:  # 长度不足进行填充
            words = (self.sequence_length - len(words)) * ['unknow'] + words
        word_ids = map(lambda word: self.words_id[word] if word in self.words_id else self.words_id['unknow'], words)
        return [word_ids]

