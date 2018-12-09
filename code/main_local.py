import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from sklearn import metrics
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import random
import time
from collections import defaultdict

class Model(object):
    def __init__(self, shape, learning_rate, latent_factors, regularizationCoeff, loss, baseline, rating_avg = 3.5, random_state = 0):
        self.shape = shape
        self.learning_rate = learning_rate
        self.latent_factors = latent_factors
        self.loss = loss
        self.random_state = random_state
        self.regularizationCoeff = regularizationCoeff

        # the R (n, i) matrix is factorized to P (n, k) and Q (i, k) matrices
        n, i, k = self.shape

        # initialize the graph
        self.graph = tf.Graph()

        with self.graph.as_default():

            tf.set_random_seed(self.random_state)
            with tf.name_scope('inputs'):

                self.user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
                self.item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
                self.ratings = tf.placeholder(tf.float32, shape=[None], name='ratings')

                targets = tf.identity(self.ratings)


            with tf.name_scope('parameters'):

                if baseline != 1:
                    # self.user_embeddings = tf.get_variable('user_embeddings', shape=[n, k], dtype=tf.float32, initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

                    # self.item_embeddings = tf.get_variable('item_embeddings', shape=[i, k], dtype=tf.float32, initializer = tf.random_normal_initializer(mean=0, stddev=0.01))
                    self.user_embeddings = tf.get_variable('user_embeddings', shape=[n, k], dtype=tf.float32, initializer = tf.zeros_initializer())

                    self.item_embeddings = tf.get_variable('item_embeddings', shape=[i, k], dtype=tf.float32, initializer = tf.zeros_initializer())

                self.user_bias = tf.get_variable('user_bias', shape=[n], dtype=tf.float32, initializer=tf.zeros_initializer())
                self.item_bias = tf.get_variable('item_bias', shape=[i], dtype=tf.float32, initializer=tf.zeros_initializer())

                # self.global_bias = tf.get_variable('global_bias', shape=[], dtype=tf.float32, initializer=tf.zeros_initializer())

            with tf.name_scope('prediction'):
                # batch
                batch_user_bias = tf.nn.embedding_lookup(self.user_bias, self.user_ids)
                batch_item_bias = tf.nn.embedding_lookup(self.item_bias, self.item_ids)

                if baseline != 1:
                    batch_user_embeddings = tf.nn.embedding_lookup(self.user_embeddings, self.user_ids)
                    batch_item_embeddings = tf.nn.embedding_lookup(self.item_embeddings, self.item_ids)
                    # P[u,:] * Q[i,:]
                    temp_sum = tf.reduce_sum(tf.multiply(batch_user_embeddings, batch_item_embeddings), axis=1)

                bias = tf.add(batch_user_bias, batch_item_bias)
                # bias = tf.add(bias, self.global_bias)
                bias = tf.add(bias, tf.constant(rating_avg))
                if baseline == 1:
                    self.pred = tf.identity(bias, name='predictions')
                else:
                    predictor = tf.add(bias, temp_sum)
                    self.pred = tf.identity(predictor, name='predictions')
                # self.pred = tf.identity(bias, name='predictions')

                # all
                # temp_sum = tf.reduce_sum(tf.multiply(self.user_embeddings, self.item_embeddings), axis=1)
                # bias = tf.add(self.user_bias, self.item_bias)
                # bias = tf.add(bias, self.global_bias)

                # predictor = tf.add(bias, temp_sum)

                # self.pred = tf.identity(predictor, name='predictions')

            with tf.name_scope('loss'):

                # l2_bias = tf.add(tf.nn.l2_loss(self.user_bias),tf.nn.l2_loss(self.item_bias))
                # l2_term = tf.identity(l2_bias)
                loss_raw = tf.losses.mean_squared_error(predictions=self.pred, labels=targets)
                if baseline != 1:
                    if (FLAGS.regularization_switch == 0):
                        self.cost = tf.identity(loss_raw)
                    elif (FLAGS.regularization_switch == 1):
                        l1_weights = tf.add(tf.reduce_sum(tf.abs(self.user_embeddings)),tf.reduce_sum(tf.abs(self.item_embeddings)))
                        l1_term = tf.multiply(self.regularizationCoeff, l1_weights, name='regularization')
                        self.cost = tf.add(loss_raw, l1_term)
                    elif (FLAGS.regularization_switch == 2):
                        l2_weights = tf.add(tf.nn.l2_loss(self.user_embeddings),tf.nn.l2_loss(self.item_embeddings))
                        # l2_term = tf.add(l2_weights, l2_bias)
                        l2_term = tf.multiply(self.regularizationCoeff, l2_weights, name='regularization')
                        self.cost = tf.add(loss_raw, l2_term)
                    else:
                        assert False
                else:
                    self.learning_rate = .08
                    self.cost = tf.identity(loss_raw)

                self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
                # self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

            init = tf.global_variables_initializer()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(init)

    def train(self, users, items, ratings):
        batch = {
                self.user_ids:users,
                self.item_ids:items,
                self.ratings:ratings
                }
        _, loss_value = self.sess.run(fetches=[self.train_step, self.cost],
                feed_dict=batch)
        return loss_value


    def predict(self, users, items):
        batch = {
                self.user_ids : users,
                self.item_ids : items
                }
        return self.pred.eval(feed_dict=batch, session=self.sess)

class MatrixFactorize(object):

    def __init__(self, latent_factors, batch_size, n_epoch, learning_rate, regularizationCoeff, baseline):
        self.latent_factors = latent_factors
        self.shape = (None, None, self.latent_factors)
        self._tf = None
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.learning_rate = float(learning_rate)
        self.regularizationCoeff = regularizationCoeff
        self.baseline = baseline
        # self.warm_start = warm_start
        self.loss = 'MSE'
        self._fresh_session()


    def _fresh_session(self):
        # reset the session, to start from the scratch        
        self._tf = None
        self.train_loss = []
        self.test_loss = []

    def init_with_shape(self, u, i):
        self.shape = (int(u), int(i), int(self.latent_factors))
        self._tf_init()

    def _tf_init(self):
        self._tf = Model(shape=self.shape, learning_rate=self.learning_rate, latent_factors=self.latent_factors, regularizationCoeff=self.regularizationCoeff, loss=self.loss, baseline=self.baseline)

    def fit(self, sparse_matrix, test=None):
        # 
        # Fit the model

        # Fit the model starting at randomly initialized parameters. When
        # warm_start=True, this method works the same as partial_fit.

        # Parameters
        # ----------

        # sparse_matrix : sparse-matrix, shape (n_users, n_items)
        # 	Sparse matrix in scipy.sparse format, can be created using sparse_matrix
        # 	function from this package.
        # 
        # if not self.warm_start:
        # 	self._fresh_session()
        if test != None:
            return self.testloss_fit(sparse_matrix, test)
        else:
            return self.partial_fit(sparse_matrix)

    def partial_fit(self, sparse_matrix):
        """Fit the model

        Fit the model starting at previously trained parameter values. If the
        model was not trained yet, it randomly initializes parameters same as
        the fit method.

        Parameters
        ----------

        sparse_matrix : sparse-matrix, shape (n_users, n_items)
            Sparse matrix in scipy.sparse format, can be created using sparse_matrix
            function from this package.
        """
        if self._tf is None:
            self.init_with_shape(*sparse_matrix.shape)

        batch = self._batch_generator(sparse_matrix, size=self.batch_size)

        # for _ in tqdm.trange(self.n_epoch):
        #     batch_users, batch_items, batch_ratings = next(batch)
        #     loss_value = self._tf.train(batch_users, batch_items, batch_ratings)
        #     self.train_loss.append(loss_value)
        loss_values = []
        users, items = sparse_matrix.nonzero()
        upto = len(users)
        for ep in range(self.n_epoch):
            indices = np.random.permutation(upto)
            step = 0
            counter = 0
            cum_loss = 0
            while step < upto:
                idx = indices[step:step+self.batch_size]
                loss_value = self._tf.train(users[idx], items[idx], sparse_matrix[users[idx], items[idx]].A.flatten())
                step += self.batch_size
                cum_loss += loss_value
                counter += 1
            print('epoch: {}, loss_value: {}'.format(ep+1, cum_loss/counter))
            loss_values.append(cum_loss/counter)
        return loss_values

    def testloss_fit(self, sparse_matrix, test):
        if self._tf is None:
            self.init_with_shape(*sparse_matrix.shape)

        batch = self._batch_generator(sparse_matrix, size=self.batch_size)

        for _ in trange(self.n_epoch):
            batch_users, batch_items, batch_ratings = next(batch)
            train_loss_value = self._tf.train(batch_users, batch_items, batch_ratings)
            self.train_loss.append(train_loss_value)

            if _ % 10 == 0:
                test_loss_value = self.cal_test(test)
                self.test_loss.append(test_loss_value)

        return self

    # def _batch_generator(self, data, size=1):
    #     # for explicit ratings
    #     users, items = data.nonzero()
    #     while True:
    #         idx = np.random.randint(len(users), size=size)
    #         yield users[idx], items[idx], data[users[idx], items[idx]].A.flatten()

    def _batch_generator(self, data, size=1):
        # for explicit ratings
        users, items = data.nonzero()
        upto = len(users)
        indices = np.random.permutation(upto)
        step = 0
        while step < upto:
            idx = indices[step:step+size]
            yield users[idx], items[idx], data[users[idx], items[idx]].A.flatten()
            step += size



    def get_losses(self):
        return self.train_loss, self.test_loss

    def predict(self, users, items):
        """Predict using the model
        """
        return self._tf.predict(users,items)

    def cal_test(self, test_matrix):
        users,items = test_matrix.nonzero()

        pred = self.predict(users, items)
        y_ = test_matrix[users, items].A.flatten()

        MSE = metrics.mean_squared_error(pred, y_)
        MAE = metrics.mean_absolute_error(pred, y_)
        print('RMSE = {}'.format(np.sqrt(MSE)))
        print('MAE = {}'.format(MAE))
        # print(pred)
        # print(y_)
        # print(MSE)

        # return MSE

def load_data(path):
    # with open(path,'r') as f:
    # 	data = f.readlines()
    # 	print(len(data))
    df = pd.read_csv(path)

    rows, user_2_id = norm(df['userId'])
    cols, movie_2_id = norm(df['movieId'])
    ratings = df['rating'].values

    n = np.max(rows) + 1
    k = np.max(cols) + 1

    coo = sparse.coo_matrix((ratings, (rows, cols)), shape=(n,k))

    return coo.todok()

def norm(series):
    index_dic = dict()
    count = 0
    for i in series.unique():
        index_dic[i] = count
        count += 1
    return series.apply(lambda x:index_dic[x]).values, index_dic


def train_test_split(path, frac=0.8, random_state=1):
    # ml-latest-small
    if 'ml-100k/ratings.csv' in path:
        df = pd.read_csv(path, sep=',', usecols=[0,1,2],dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names = ['userId','movieId','rating'], engine='python', header=None, skiprows=1)
    elif 'ml-1m/ratings.dat' in path:
        # ml-1m
        df = pd.read_csv(path, sep='::', usecols=[0,1,2],dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names = ['userId','movieId','rating'], engine='python', header=None)
    elif 'ml-10m/ratings.dat' in path:
        # ml-1m
        df = pd.read_csv(path, sep='::', usecols=[0,1,2],dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names = ['userId','movieId','rating'], engine='python', header=None)
    elif 'ml-20m/ratings.csv' in path:
        # ml-20m
        df = pd.read_csv(path, sep=',', usecols=[0,1,2], dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names=['userId','movieId','rating'], engine='python', skiprows=1)
    elif 'ml-100m/ratings.csv' in path:
        # ml-100m NetFlix
        df = pd.read_csv(path, sep='\t', dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names=['userId','movieId','rating'], usecols = ["userId", "moviedId", "rating"], engine='python', skiprows=1)
    else:
        raise ValueError("Can't figure out if", path, 'is 20M or 1M or 100K MovieLens DataSet')

    users, user_2_id = norm(df['userId'])
    movies, movie_2_id = norm(df['movieId'])
    ratings = df['rating'].values

    n = np.max(users) + 1
    k = np.max(movies) + 1

    train_idx = df.sample(frac=frac,random_state=random_state).index.values
    # train_idx = df.sample(frac=frac).index.values
    test_idx = test_idx = np.array(list(set(range(len(ratings))) - set(train_idx)))

    train_data = sparse.coo_matrix((ratings[train_idx], (users[train_idx], movies[train_idx])), shape=(n,k))
    test_data = sparse.coo_matrix((ratings[test_idx], (users[test_idx], movies[test_idx])), shape=(n,k))

    return train_data.todok(), test_data.todok(), ''


def record(x,dic):
    dic[x.userId].append(x.movieId)
    return 1

def get_interaction(df):
    dic = defaultdict(list)
    a = df.apply(lambda x:record(x,dic),axis=1)
    return dic

def get_input_array(positive, negative, user_2_id, movie_2_id,train=False):
    users = []
    movies = []
    ratings = []

    for user in positive:
        for index in range(len(positive[user])):
            users.append(user_2_id[user])
            users.append(user_2_id[user])
            movies.append(movie_2_id[positive[user][index]])
            movies.append(movie_2_id[negative[user][index]])
            if train:
                ratings.append(1)
                ratings.append(0)
            else:
                ratings.append(2)
                ratings.append(1)
    return np.array(users), np.array(movies), np.array(ratings)
# from mycf.gcn_mf import GCNMF
# from mycf.cf_model import CFModel
# from mycf import GCNMF, sparse_matrix

# user_id = [0,0,1,1,2,2]
# movie_id = [0,1,2,0,1,2]
# rating = [1,1,2,2,3,3]
# X = sparse_matrix(user_id, movie_id, rating)

# X = load_data('data/ml-latest-small/ratings.csv')
# data/ml-1m/ratings.dat

# Define parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_path", "", "FilePath of MovieLens Dataset")
tf.app.flags.DEFINE_integer("latent_factors", 5, "# of Latent Factors")
tf.app.flags.DEFINE_integer("batch_size", 500, "Batch Size for Training")
tf.app.flags.DEFINE_integer("n_epoch", 20, "# of Epochs")
tf.app.flags.DEFINE_float('lr', 0.001, 'Learning rate')
tf.app.flags.DEFINE_float('regularizationCoeff', 0.02, 'Regularization Coefficient')
tf.app.flags.DEFINE_integer('steps_to_validate', 100, 'Step to validate and print loss')
tf.app.flags.DEFINE_integer('baseline', 0, 'Run for baseline')
tf.app.flags.DEFINE_integer('regularization_switch', 2, 'Whether 0/1/2 2 mean L2, 1 mean L1 and 0 means no regularization is applied')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-seperated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")

def main(_):
    if FLAGS.data_path  == "":
        raise ValueError("Must specify an explicit `data_path`")

    # ps_hosts = FLAGS.ps_hosts.split(",")
    # worker_hosts = FLAGS.worker_hosts.split(",")
    # cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    print('Using the following : data_path =', FLAGS.data_path, ', latent_factors =', FLAGS.latent_factors, ', batch_size =', FLAGS.batch_size, ', n_epochs =', FLAGS.n_epoch, ', Initial Learning Rate =', FLAGS.lr, ', regularizationCoeff =', FLAGS.regularizationCoeff, 'baseline =', FLAGS.baseline, 'regularization_switch =', FLAGS.regularization_switch)
    # print('flags = ', FLAGS)

    print('Loading data...')
    t1 = time.time()
    train_data, test_data, user_idx = train_test_split(FLAGS.data_path, frac=0.8,random_state=1)
    t2 = time.time()
    print('Loading and pre-processing time = {} seconds'.format(t2-t1))
    # print(train_data.toarray())
    # print(test_data.toarray())
    print('Training Epochs :')
    t1 = time.time()
    mf = MatrixFactorize(FLAGS.latent_factors, FLAGS.batch_size, FLAGS.n_epoch, FLAGS.lr, FLAGS.regularizationCoeff, FLAGS.baseline)
    losses = mf.fit(train_data)
    t2 = time.time()
    print('Training time = {} seconds'.format(t2-t1))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x_values = [int(i) for i in range(1, FLAGS.n_epoch + 1)]
    ax.plot(x_values, losses, '--')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss vs Epochs')
    plt.show()
    print('Testing on the training set')
    t1 = time.time()
    mf.cal_test(train_data)
    t2 = time.time()
    print('Testing time on Training dataset = {} seconds'.format(t2-t1))
    print('Testing on the test set')
    t1 = time.time()
    mf.cal_test(test_data)
    t2 = time.time()
    print('Testing time on Test dataset = {} seconds'.format(t2-t1))
# print(loss)
# mf.fit(train_data, test_data)
# train_loss, test_loss = mf.get_losses()
# print(train_loss)
# print(test_loss)
# print(mf.predict_all().A)
# mf.print_history()
if __name__ == "__main__":
    tf.app.run()
