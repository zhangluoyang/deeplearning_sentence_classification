# coding=utf-8
"""
深度学习应用于句子的情感分析
2017-03-11　张洛阳　南京
"""
import tensorflow as tf
import logging
import os
import numpy as np
"""
Kim Y. Convolutional neural networks for sentence classification[J]. arXiv preprint arXiv:1408.5882, 2014.
"""
class single_random_model():
    """
    卷积神经网络单通道  word embedding 采用的是随机初始化
    """
    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, model_name="single_random_model"):
        """
        :param sequence_length:  句子的长度  int
        :param num_classes:  类别数目  int
        :param vocab_size:  训练集当中总词数  int
        :param embedding_size:  词嵌入维数的大小  int
        :param filter_sizes:  采用的卷积和的尺寸  list
        :param num_filters:  滤波器的数目 int
        :param l2_reg_lambda:  l2 正则化参数
        """
        self.model_name = model_name
        # [batch_size, sequence_length]
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # [batch_size, num_classes]
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        # 词的嵌入维数层
        with tf.name_scope("embedding"):
            # [vocab_size, embedding_size]
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            # [batch_size, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # [batch_size, sequence_length, embedding_size, 1]  二维卷积的形式
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积非线性变换+池化
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        # [ [batch_size, sentence_length, 1, num_filters], .... ] list的长度是
        h_pool = tf.concat(3, pooled_outputs)
        # [batch_size, num_filters_total]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            # [num_filters_total, num_classes]
            W = tf.Variable(tf.random_uniform([num_filters_total, num_classes]), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            xWb = tf.nn.xw_plus_b(h_drop, W, b)  # 类别的得分
            self.scores = tf.nn.softmax(xWb, name="scores")
            self.predictions = tf.argmax(xWb, 1, name="predictions")  # 具体预测值
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_y)  # 交叉熵代价函数
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")  # 准确度

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())
            def train_step(x_batch, y_batch):
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))
            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: 1.0}
            accuracys = self.sess.run([self.accuracys], feed_dict)
            return accuracys

        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print("mean acc:{}".format(np.mean(np.array(eval_acc))))

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_x: input_x, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores

class single_static_model():
    def __init__(self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0, model_name="single_static_model"):
        """
        :param sequence_length:  句子的长度  int
        :param num_classes:  类别数目  int
        :param vocab_size:  训练集当中总词数  int
        :param embedding_size:  词嵌入维数的大小  int
        :param filter_sizes:  采用的卷积和的尺寸  list
        :param num_filters:  滤波器的数目 int
        :param l2_reg_lambda:  l2 正则化参数
        """
        self.model_name = model_name
        # [batch_size, sequence_length, embedding_size]
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        with tf.name_scope("process"):
            # [batch_size, sequence_length, embedding_size, 1]
            input_x_c = tf.expand_dims(self.input_x, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积非线性变换+池化
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(input_x_c, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool( h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        # [ [batch_size, sentence_length, 1, num_filters], .... ] list的长度是
        h_pool = tf.concat(3, pooled_outputs)
        # [batch_size, num_filters_total]
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            xWb = tf.nn.xw_plus_b(h_drop, W, b)  # 类别的得分
            self.scores = tf.nn.softmax(xWb, name="scores")
            self.predictions = tf.argmax(xWb, 1, name="predictions")
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_y)  # 交叉熵代价函数
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")  # 准确度

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))

            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: 1.0}
            accuracys = self.sess.run([self.accuracys], feed_dict)
            return accuracys

        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print("mean acc:{}".format(np.mean(np.array(eval_acc))))

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_x: input_x, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores

class single_truned_model():
    def __init__(
            self, sequence_length, num_classes, embedding_size, filter_sizes, num_filters, word2vec_init, l2_reg_lambda=0.0,
            model_name="single_truned_model"):
        """
        :param sequence_length:  句子的长度  int
        :param num_classes:  类别数目  int
        :param vocab_size:  训练集当中总词数  int
        :param embedding_size:  词嵌入维数的大小  int
        :param filter_sizes:  采用的卷积和的尺寸  list
        :param num_filters:  滤波器的数目 int
        :param l2_reg_lambda:  l2 正则化参数
        """
        self.model_name = model_name
        # [batch_size, sequence_length]
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        # [batch_size, num_classes]
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        with tf.name_scope("embedding"):
            # [len(word2vec), embedding_size]
            W = tf.Variable(word2vec_init, name="W")
            # [batch_size, sequence_length, embedding_size]
            embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            # [batch_size, sequence_length, embedding_size, 1]
            embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        num_filters_total = num_filters * len(filter_sizes)
        # [ [batch_size, sentence_length, 1, num_filters], .... ] list的长度是
        h_pool = tf.concat(3, pooled_outputs)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            xWb = tf.nn.xw_plus_b(h_drop, W, b)
            self.scores = tf.nn.softmax(xWb, name="scores")
            self.predictions = tf.argmax(xWb, 1, name="predictions")
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")

    def fit(self, datas, epochs, keep_prob=0.5):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a",
                            level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: keep_prob}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))

            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch, self.dropout_keep_prob: 1.0}
            accuracys = self.sess.run([self.accuracys], feed_dict)
            return accuracys

        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print("mean acc:{}".format(np.mean(np.array(eval_acc))))

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_x: input_x, self.dropout_keep_prob: 1.0}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores

"""
LSTM Networks for Sentiment Analysis
http://deeplearning.net/tutorial/lstm.html
"""
class lstm_mean_model(object):
    def __init__(self, sequence_length, class_num=3, embedding_size=200, l2_reg_lambda=0.0, model_name="lstm_mean_model"):
        """

        :param sequence_length: 序列长度
        :param class_num: 类别数目
        :param embedding_size: 词嵌入维数
        :param l2_reg_lambda: 正则化参数
        """
        self.model_name = model_name
        # [batch_size, sequence_length, embedding_size]
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder('float', [None, class_num], name="input_y")
        with tf.name_scope('process'):
            # [sequence_length, batch_size, embedding_size]
            input_xt = tf.transpose(self.input_x, [1, 0, 2])
            # [seqence_length*batch_size, embedding_size]
            input_xr = tf.reshape(input_xt, [-1, embedding_size])
            input_split = tf.split(0, sequence_length, input_xr)
        l2_loss = tf.constant(0.0)
        with tf.name_scope("lstm"):
            lstm = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            outputs, _states = tf.nn.rnn(lstm, input_split, dtype=tf.float32)
        with tf.name_scope("output"):
            W = tf.get_variable(name="W", shape=(embedding_size, class_num))
            b = tf.get_variable(name="b", shape=(class_num))
            mean_pool = tf.add_n(outputs) / sequence_length
            xWb = tf.nn.xw_plus_b(mean_pool, W, b)
            self.scores = tf.nn.softmax(xWb, name="scores")
            self.predictions = tf.argmax(xWb, 1, name="predictions")
        with tf.name_scope("loss"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(xWb, self.input_y), name="cost")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.loss = cost + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")
    def fit(self, datas, epochs):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a", level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())
            def train_step(x_batch, y_batch):
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))
            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch}
            accuracys = self.sess.run([self.accuracys], feed_dict)
            return accuracys
        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print("mean acc:{}".format(np.mean(np.array(eval_acc))))

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_x: input_x}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores
"""
Word2vecSentimentRNN
https://github.com/deeplearning4j/dl4j-examples/tree/master/dl4j-examples/src/main/java/org/deeplearning4j/examples/recurrent/word2vecsentiment
"""
class bi_lstm_model(object):
    def __init__(self, seq_length, class_num=3, embedding_size=200, l2_reg_lambda=0.0, model_name="bi_lstm_model"):
        """
        :param seq_length: 句子的长度
        :param class_num: 分类的类别
        :param embedding_size: 词嵌入维数
        :param l2_reg_lambda:
        """
        self.model_name = model_name
        # input_x [batch_size,seq_length,embedding_size]
        self.input_x = tf.placeholder(tf.float32, [None, seq_length, embedding_size], name="input_x")
        self.input_y = tf.placeholder('float', [None, class_num], name="input_y")
        with tf.name_scope("process"):
            # input_xt [seq_length,batch_size,embedding_size]
            input_xt = tf.transpose(self.input_x, [1, 0, 2])
            # input_xr [](seq_length*batch_size,embedding_size)
            input_xr = tf.reshape(input_xt, [-1, embedding_size])
            # [(batch_size,embedding_size),.........................]
            input_split = tf.split(0, seq_length, input_xr)
            l2_loss = tf.constant(0.0)
        with tf.name_scope("lstm"):
            self.lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            self.lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(embedding_size, forget_bias=1.0)
            outputs, _, _ = tf.nn.bidirectional_rnn(self.lstm_fw_cell, self.lstm_bw_cell, input_split, dtype=tf.float32)
        with tf.name_scope("output"):
            W = tf.Variable(tf.random_normal([2 * embedding_size, class_num], stddev=0.01), name="W")
            b = tf.Variable(tf.random_normal([class_num]), name="b")
            self.scores = tf.nn.xw_plus_b(outputs[-1], W, b, name="scores")
            self.scores_scale = tf.nn.softmax(self.scores, name="scores_scale")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        with tf.name_scope("loss"):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y), name="cost")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.loss = cost + l2_reg_lambda * l2_loss
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracys = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracys")

    def fit(self, datas, epochs):
        import datetime
        logging.basicConfig(filename="{}.log".format(self.model_name), format="%(message)s", filemode="a", level=logging.INFO)
        checkpoint_dir = os.path.abspath(os.path.join(self.model_name, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        with tf.Session() as sess:
            self.sess = sess
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(self.loss)
            train_op = optimizer.apply_gradients(grads_and_vars)
            saver = tf.train.Saver(tf.all_variables(), max_to_keep=10)
            sess.run(tf.initialize_all_variables())
            def train_step(x_batch, y_batch):
                feed_dict = {self.input_x: x_batch, self.input_y: y_batch}
                _, loss, accuracys = sess.run([train_op, self.loss, self.accuracys], feed_dict)
                time_str = datetime.datetime.now().isoformat()
                logging.info("{}: loss: {}, acc: {}".format(time_str, loss, accuracys))
            for i in range(epochs):
                datas.shuffle()
                for _ in range(datas.batch_nums):
                    x_batch, y_batch = datas.next_batch()
                    train_step(x_batch, y_batch)
                saver.save(sess, checkpoint_prefix, global_step=i)  # 存储训练数据

    def eval(self, datas):
        def eval_step(x_batch, y_batch):
            feed_dict = {self.input_x: x_batch, self.input_y: y_batch}
            accuracys = self.sess.run([self.accuracys], feed_dict)
            return accuracys
        eval_acc = []
        for _ in range(datas.batch_nums):
            x_batch, y_batch = datas.next_batch()
            acc = eval_step(x_batch=x_batch, y_batch=y_batch)
            eval_acc.append(acc)
        print("mean acc:{}".format(np.mean(np.array(eval_acc))))

    def load(self, checkpointfile):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            self.sess = sess
            saver = tf.train.import_meta_graph("{}.meta".format(checkpointfile))
            saver.restore(sess, checkpointfile)
            self.input_x = graph.get_operation_by_name("input_x").outputs[0]
            self.input_y = graph.get_operation_by_name("input_y").outputs[0]
            self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            self.accuracys = graph.get_operation_by_name("accuracy/accuracys").outputs[0]
            self.scores = graph.get_operation_by_name("output/scores").outputs[0]

    def predict(self, input_x):
        feed_dict = {self.input_x: input_x}
        predictions, scores = self.sess.run([self.predictions, self.scores], feed_dict=feed_dict)
        return predictions, scores