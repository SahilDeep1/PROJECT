import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
from sklearn import metrics
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import random
import math
import time
from collections import defaultdict


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
tf.app.flags.DEFINE_integer('baseline', 0, 'Run for baseline')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-seperated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")

def main(_):
    if FLAGS.data_path  == "":
        raise ValueError("Must specify an explicit `data_path`")

    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    job_name = FLAGS.job_name
    task_index = FLAGS.task_index
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)
    if job_name == 'ps':
        print('PS started')
        server.join()
    elif job_name == 'worker':
        print('This is a worker node')

        print('Using the following : data_path =', FLAGS.data_path, ', latent_factors =', FLAGS.latent_factors, ', batch_size =', FLAGS.batch_size, ', n_epochs =', FLAGS.n_epoch, ', Initial Learning Rate =', FLAGS.lr, ', regularizationCoeff =', FLAGS.regularizationCoeff, 'baseline =', FLAGS.baseline)
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
        shape = (*train_data.shape, FLAGS.latent_factors)
        # print('Shape = ', shape)
        n, i , k = shape

        with tf.device(tf.train.replica_device_setter(worker_device="/job:worker/replica:0/task:{}".format(task_index), cluster=cluster)):
            # global_step coordinates between all workers the current step of training
            # global_step = tf.contrib.framework.get_or_create_global_step()
            global_step = tf.train.get_or_create_global_step()
            # More to come on is_chief...
            # is_chief = (task_index == 0)

            tf.set_random_seed(0)
            # with tf.name_scope('inputs'):

            user_ids = tf.placeholder(tf.int32, shape=[None], name='user_ids')
            item_ids = tf.placeholder(tf.int32, shape=[None], name='item_ids')
            ratings = tf.placeholder(tf.float32, shape=[None], name='ratings')

            targets = tf.identity(ratings)


            # with tf.name_scope('parameters'):

            if FLAGS.baseline != 1:
                # self.user_embeddings = tf.get_variable('user_embeddings', shape=[n, k], dtype=tf.float32, initializer = tf.random_normal_initializer(mean=0, stddev=0.01))

                # self.item_embeddings = tf.get_variable('item_embeddings', shape=[i, k], dtype=tf.float32, initializer = tf.random_normal_initializer(mean=0, stddev=0.01))
                user_embeddings = tf.get_variable('user_embeddings', shape=[n, k], dtype=tf.float32, initializer = tf.zeros_initializer())

                item_embeddings = tf.get_variable('item_embeddings', shape=[i, k], dtype=tf.float32, initializer = tf.zeros_initializer())

            user_bias = tf.get_variable('user_bias', shape=[n], dtype=tf.float32, initializer=tf.zeros_initializer())
            item_bias = tf.get_variable('item_bias', shape=[i], dtype=tf.float32, initializer=tf.zeros_initializer())

                # self.global_bias = tf.get_variable('global_bias', shape=[], dtype=tf.float32,
                #         initializer=tf.zeros_initializer())

            # with tf.name_scope('prediction'):
                # batch
            batch_user_bias = tf.nn.embedding_lookup(user_bias, user_ids)
            batch_item_bias = tf.nn.embedding_lookup(item_bias, item_ids)

            if FLAGS.baseline != 1:
                batch_user_embeddings = tf.nn.embedding_lookup(user_embeddings, user_ids)
                batch_item_embeddings = tf.nn.embedding_lookup(item_embeddings, item_ids)
                # P[u,:] * Q[i,:]
                temp_sum = tf.reduce_sum(tf.multiply(batch_user_embeddings, batch_item_embeddings), axis=1)

            bias = tf.add(batch_user_bias, batch_item_bias)
            # bias = tf.add(bias, global_bias)
            bias = tf.add(bias, tf.constant(3.5))
            if FLAGS.baseline == 1:
                pred = tf.identity(bias, name='predictions')
            else:
                predictor = tf.add(bias, temp_sum)
                pred = tf.identity(predictor, name='predictions')
                # pred = tf.identity(bias, name='predictions')

                # all
                # temp_sum = tf.reduce_sum(tf.multiply(user_embeddings, item_embeddings), axis=1)
                # bias = tf.add(user_bias, item_bias)
                # bias = tf.add(bias, global_bias)

                # predictor = tf.add(bias, temp_sum)

                # pred = tf.identity(predictor, name='predictions')

            # with tf.name_scope('loss'):

                # l2_bias = tf.add(tf.nn.l2_loss(user_bias),tf.nn.l2_loss(item_bias))
                # l2_term = tf.identity(l2_bias)
            loss_raw = tf.losses.mean_squared_error(predictions=pred, labels=targets)
            if FLAGS.baseline != 1:
                l2_weights = tf.add(tf.nn.l2_loss(user_embeddings),tf.nn.l2_loss(item_embeddings))
                # l2_term = tf.add(l2_weights, l2_bias)
                l2_term = tf.multiply(FLAGS.regularizationCoeff, l2_weights, name='regularization')
                cost = tf.add(loss_raw, l2_term)
            else:
                learning_rate = .08
                cost = tf.identity(loss_raw)


            # train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
            # train_step = tf.train.GradientDescentOptimizer(FLAGS.lr).minimize(cost, global_step=global_step)
            train_step = tf.train.AdamOptimizer(FLAGS.lr).minimize(cost, global_step=global_step)
            #saver allows for saving/restoring variables to/from checkpoints during training
            # saver = tf.train.Saver()
            #summary_op tracks all summaries of the graph
            # summary_op = tf.summary.merge_all()
            #init_op defines the operation to initialize all tf.Variable()s
            # init_op = tf.global_variables_initializer()
        #print('Created Model, fitting data now')

        users, items = train_data.nonzero()
        upto = len(users)
        iterations_per_epoch = math.ceil(upto/FLAGS.batch_size)
        last_step = FLAGS.n_epoch * iterations_per_epoch
        print('records =', upto, 'iterations_per_epoch =', iterations_per_epoch, 'last_step = ', last_step)
        # hooks=[tf.train.StopAtStepHook(last_step=last_step)]
        # with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="./train_logs", hooks=hooks) as mon_sess:
        # with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0), checkpoint_dir="./train_logs") as mon_sess:
        with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0)) as mon_sess:
            loss_values = []
            ep = 0
            for ep in range(FLAGS.n_epoch):
                indices = np.random.permutation(upto)
                frm = 0
                cum_loss = 0
                counter = 0
                for it in range(iterations_per_epoch):
                    idx = indices[frm:frm+FLAGS.batch_size]
                    batch = {
                            user_ids:users[idx],
                            item_ids:items[idx],
                            ratings:train_data[users[idx], items[idx]].A.flatten()
                            }
                    _, loss_value  = mon_sess.run([train_step, cost], feed_dict=batch)
                    frm += FLAGS.batch_size
                    cum_loss += loss_value
                    counter += 1
                assert counter == iterations_per_epoch
                avg_epoch_loss = cum_loss/counter
                    # train_accuracy = mon_sess.run(accuracy, feed_dict={ x: batch[0], y_: batch[1], keep_prob: 1.0})
                print('global_step {}, task:{}, epoch:{}, loss:{} '.format(tf.train.global_step(mon_sess, global_step), FLAGS.task_index, ep+1, avg_epoch_loss))
                loss_values.append(avg_epoch_loss)
            t2 = time.time()
            print('Training time = {} seconds'.format(t2-t1))

            # print('Fitted Data, plotting loss now')
            fig = plt.figure()
            ax = fig.add_subplot(111)
            x_values = [i for i in range(1, FLAGS.n_epoch + 1)]
            ax.plot(x_values, loss_values, '--')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Training Loss')
            ax.set_title('Training Loss vs Epochs')
            plt.show()
            print('Testing on the training set')
            t1 = time.time()
            users,items = train_data.nonzero()
            batch = {
                    user_ids : users,
                    item_ids : items
                    }
            predicted_ratings = pred.eval(feed_dict=batch, session=mon_sess)
            true_ratings = train_data[users, items].A.flatten()

            MSE = metrics.mean_squared_error(predicted_ratings, true_ratings)
            MAE = metrics.mean_absolute_error(predicted_ratings, true_ratings)
            print('RMSE = {}'.format(np.sqrt(MSE)))
            print('MAE = {}'.format(MAE))
            t2 = time.time()
            print('Testing time on Training dataset = {} seconds'.format(t2-t1))


            print('Testing on the test set')
            t1 = time.time()
            users,items = test_data.nonzero()

            batch = {
                    user_ids : users,
                    item_ids : items
                    }
            predicted_ratings = pred.eval(feed_dict=batch, session=mon_sess)
            true_ratings = test_data[users, items].A.flatten()

            MSE = metrics.mean_squared_error(predicted_ratings, true_ratings)
            MAE = metrics.mean_absolute_error(predicted_ratings, true_ratings)
            print('RMSE = {}'.format(np.sqrt(MSE)))
            print('MAE = {}'.format(MAE))
            t2 = time.time()
            print('Testing time on test dataset = {} seconds'.format(t2-t1))
        # print('Testing on the test set'
        # mf.cal_test(test_data,user_idx)
# print(loss)
# mf.fit(train_data, test_data)
# train_loss, test_loss = mf.get_losses()
# print(train_loss)
# print(test_loss)
# print(mf.predict_all().A)
# mf.print_history()
if __name__ == "__main__":
    tf.app.run()
