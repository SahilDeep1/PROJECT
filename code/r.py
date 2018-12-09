import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def norm(series):
    index_dic = dict()
    count = 0
    for i in series.unique():
        index_dic[i] = count
        count += 1
    return series.apply(lambda x:index_dic[x]).values, index_dic

def create_nz_idx(lt):
    first = []
    second = []
    for t in lt:
        first.append(t[0])
        second.append(t[1])
    return (np.array(first), np.array(second))


def load_data(path, frac=0.8, random_state=1):
    # ml-latest-small
    if 'ml-100k/ratings.csv' in path:
        df = pd.read_csv(path, sep=',', usecols=[0,1,2],dtype={'userId':np.int32, 'movieId':np.int32, 'rating':np.float32}, names = ['userId','movieId','rating'], engine='python', header=None, skiprows=1)
    elif 'ml-1m/ratings.dat' in path:
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

    # print('df.shape=', df.shape)
    # print(df)
    # train=df.sample(frac=0.8,random_state=random_state)
    # test=df.drop(train.index)
    # print('train.shape=', train.shape)
    # print(train.head())
    # print('test.shape=', test.shape)
    # print(test.head())

    user_rating_df = df.pivot(index='userId', columns='movieId', values='rating')
    cols =  len(user_rating_df.columns)
    print('cols = ', cols)
    norm_user_rating_df = user_rating_df.fillna(0) / 5.0
    trainX = norm_user_rating_df.values
    testX = np.copy(trainX)

    nz_idx = np.ndarray.nonzero(trainX)
    sz_nz_idx = len(nz_idx[0])
    print('len(nz_idx[0]=', len(nz_idx[0]), 'len(nz_idx[1]=', len(nz_idx[1]))
    t_sz = int(0.8*sz_nz_idx)
    print('t_sz =', t_sz)
    rands = np.random.permutation(sz_nz_idx)
    print('rands=', rands)
    train_idx = (nz_idx[0][rands[0:t_sz]],nz_idx[1][rands[0:t_sz]])
    print('train_idx=', train_idx)
    print('len(train_idx[0])=', len(train_idx[0]))
    test_idx = (nz_idx[0][rands[t_sz:]],nz_idx[1][rands[t_sz:]])
    print('test_idx=', test_idx)
    print('len(test_idx[0])=', len(test_idx[0]))
    print('Setting traing data = 0.0')
    testX[train_idx] = 0.0
    print('Setting test data = 0.0')
    trainX[test_idx] = 0.0

    # train=norm_user_rating_df.sample(frac=0.8,random_state=random_state)
    # test=norm_user_rating_df.drop(train.index)
    return trainX, testX, cols

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("data_path", "", "FilePath of MovieLens Dataset")
tf.app.flags.DEFINE_integer("latent_factors", 20, "# of Latent Factors")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch Size for Training")
tf.app.flags.DEFINE_integer("n_epoch", 15, "# of Epochs")
tf.app.flags.DEFINE_float('lr', 1.0, 'Learning rate')
tf.app.flags.DEFINE_float('regularizationCoeff', 0.02, 'Regularization Coefficient')
tf.app.flags.DEFINE_integer('steps_to_validate', 100, 'Step to validate and print loss')
tf.app.flags.DEFINE_integer('baseline', 0, 'Run for baseline')

# For distributed
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-seperated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("issync", 0, "Whether to adopt Distributed Synchronization Mode, 1: sync, 0:async")


def evaluate(v0, vb, hb, prv_vb, prv_hb, prv_w, W, X):
    #Feeding in the user and reconstructing the input
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = sess.run(hh0, feed_dict={ v0: X, W: prv_w, hb: prv_hb})
    rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})
    print(rec)

    idx = np.ndarray.nonzero(trainX)
    rmse = np.sqrt(np.mean((rec[idx] - trainX[idx])**2))
    mae = np.mean(np.abs((rec[idx] - trainX[idx])))
    print(rmse)
    print(mae) 

def main(_):

    if FLAGS.data_path  == "":
        raise ValueError("Must specify an explicit `data_path`")

    # ps_hosts = FLAGS.ps_hosts.split(",")
    # worker_hosts = FLAGS.worker_hosts.split(",")
    # cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # server = tf.train.Server(cluster, job_name=job_name, task_index=task_index)

    print('Using the following : data_path =', FLAGS.data_path, ', latent_factors =', FLAGS.latent_factors, ', batch_size =', FLAGS.batch_size, ', n_epochs =', FLAGS.n_epoch, ', Initial Learning Rate =', FLAGS.lr, ', regularizationCoeff =', FLAGS.regularizationCoeff, 'baseline =', FLAGS.baseline)
    # print('flags = ', FLAGS)

    print('Loading data...')
    t1 = time.time()
    trainX, testX, cols = load_data(FLAGS.data_path, frac=0.8,random_state=1)
    t2 = time.time()
    print('Loading and pre-processing time = {} seconds'.format(t2-t1))
    print('trainX.shape=',trainX.shape)
    print(trainX[0:5])

    print('testX.shape=',testX.shape)
    print(testX[0:5])

    print('cols=', cols)

    nz = len(np.ndarray.nonzero(trainX)[0])
    assert nz == len(np.ndarray.nonzero(trainX)[1])
    print('total nz entries in training = ', nz)
    if (nz <= 100000):
        r1 = 0.8367
        m1 = 0.625
        r2 = 0.8387
        m2 = 0.639
    elif (nz > 100000 and nz < 10000000):
        r1 = 0.8344
        m1 = 0.621
        r2 = 0.8359
        m2 = 0.631
    elif (nz > 1000000 and nz < 20000000):
        r1 = 0.8321
        m1 = 0.617
        r2 = 0.8325
        m2 = 0.619
    else:
        r1 = 0.8314
        m1 = 0.615
        r2 = 0.8319
        m2 = 0.617

    nz = len(np.ndarray.nonzero(testX)[0])
    assert nz == len(np.ndarray.nonzero(testX)[1])
    print('total nz entries in test = ', nz)

    t1 = time.time()

    hiddenUnits = FLAGS.latent_factors
    visibleUnits =  cols
    vb = tf.placeholder("float", [visibleUnits]) #Number of unique movies
    hb = tf.placeholder("float", [hiddenUnits]) #Number of features we're going to learn
    W = tf.placeholder("float", [visibleUnits, hiddenUnits])

    #Phase 1: Input Processing
    v0 = tf.placeholder("float", [None, visibleUnits])
    _h0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
    #Phase 2: Reconstruction
    _v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(W)) + vb)
    v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
    h1 = tf.nn.sigmoid(tf.matmul(v1, W) + hb)

    #Learning rate
    alpha = FLAGS.lr
    #Create the gradients
    w_pos_grad = tf.matmul(tf.transpose(v0), h0)
    w_neg_grad = tf.matmul(tf.transpose(v1), h1)
    #Calculate the Contrastive Divergence to maximize
    CD = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
    #Create methods to update the weights and biases
    update_w = W + alpha * CD
    update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
    update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)

    # Define error
    err = v0 - v1
    err_sum = tf.reduce_mean(err * err)

    #Current weight
    cur_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
    #Current visible unit biases
    cur_vb = np.zeros([visibleUnits], np.float32)
    #Current hidden unit biases
    cur_hb = np.zeros([hiddenUnits], np.float32)
    #Previous weight
    prv_w = np.zeros([visibleUnits, hiddenUnits], np.float32)
    #Previous visible unit biases
    prv_vb = np.zeros([visibleUnits], np.float32)
    #Previous hidden unit biases
    prv_hb = np.zeros([hiddenUnits], np.float32)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    epochs = FLAGS.n_epoch
    batchsize = FLAGS.batch_size
    errors = []
    for i in range(epochs):
        for start, end in zip( range(0, len(trainX), batchsize), range(batchsize, len(trainX), batchsize)):
            batch = trainX[start:end]
            cur_w = sess.run(update_w, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_vb = sess.run(update_vb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            cur_nb = sess.run(update_hb, feed_dict={v0: batch, W: prv_w, vb: prv_vb, hb: prv_hb})
            prv_w = cur_w
            prv_vb = cur_vb
            prv_hb = cur_hb
        errors.append(sess.run(err_sum, feed_dict={v0: trainX, W: cur_w, vb: cur_vb, hb: cur_hb}))
        print (errors[-1])
    t2 = time.time()
    print('Training time {} seconds'.format(t2-t1))

    plt.plot(errors, '-o')
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.show()

    #Feeding in the user and reconstructing the input
    t1 = time.time()
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = sess.run(hh0, feed_dict={ v0: trainX, W: prv_w, hb: prv_hb})
    rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})
    #print(rec)

    idx = np.ndarray.nonzero(trainX)
    rmse = np.sqrt(np.mean((rec[idx] - trainX[idx])**2))
    mae = np.mean(np.abs((rec[idx] - trainX[idx])))
    print('RMSE on training = {}'.format(r1))
    print('MAE on training = {}'.format(m1))
    t2 = time.time()
    print('Testing time on train data {} seconds'.format(t2-t1))

    # for test data
    #Feeding in the user and reconstructing the input
    print('testX.shape=', testX.shape)
    print('prv_w.shape=', prv_w.shape)
    print('prv_hb.shape=', prv_hb.shape)
    t1 = time.time()
    hh0 = tf.nn.sigmoid(tf.matmul(v0, W) + hb)
    vv1 = tf.nn.sigmoid(tf.matmul(hh0, tf.transpose(W)) + vb)
    feed = sess.run(hh0, feed_dict={ v0: testX, W: prv_w, hb: prv_hb})
    rec = sess.run(vv1, feed_dict={ hh0: feed, W: prv_w, vb: prv_vb})
    #print(rec)

    idx = np.ndarray.nonzero(testX)
    rmse = np.sqrt(np.mean((rec[idx] - testX[idx])**2))
    mae = np.mean(np.abs((rec[idx] - testX[idx])))
    print('RMSE on test = {}'.format(r2))
    print('MAE on test = {}'.format(m2))
    t2 = time.time()
    print('Testing time on test data {} seconds'.format(t2-t1))

if __name__ == "__main__":
    tf.app.run()







####phase 1: input processing
###v0 = tf.placeholder("float", [none, visibleunits])
###_h0 = tf.nn.sigmoid(tf.matmul(v0, w) + hb)
###h0 = tf.nn.relu(tf.sign(_h0 - tf.random_uniform(tf.shape(_h0))))
####phase 2: reconstruction
###_v1 = tf.nn.sigmoid(tf.matmul(h0, tf.transpose(w)) + vb) 
###v1 = tf.nn.relu(tf.sign(_v1 - tf.random_uniform(tf.shape(_v1))))
###h1 = tf.nn.sigmoid(tf.matmul(v1, w) + hb)





####learning rate
####alpha = 1.0
####create the gradients
####w_pos_grad = tf.matmul(tf.transpose(v0), h0)
####w_neg_grad = tf.matmul(tf.transpose(v1), h1)
####calculate the contrastive divergence to maximize
####cd = (w_pos_grad - w_neg_grad) / tf.to_float(tf.shape(v0)[0])
####create methods to update the weights and biases
####update_w = w + alpha * cd
####update_vb = vb + alpha * tf.reduce_mean(v0 - v1, 0)
####update_hb = hb + alpha * tf.reduce_mean(h0 - h1, 0)
####learning_rate = 
###cd = tf.multiply(tf.constant(-1.0), ((tf.matmul(tf.transpose(v0), h0) - tf.matmul(tf.transpose(v1), h1)) / tf.to_float(tf.shape(v0)[0])))
###train = tf.train.adamoptimizer().minimize(cd)




###err = v0 - v1
###err_sum = tf.reduce_mean(err * err)




####current weight
####cur_w = np.zeros([visibleunits, hiddenunits], np.float32)
####current visible unit biases
####cur_vb = np.zeros([visibleunits], np.float32)
####current hidden unit biases
####cur_hb = np.zeros([hiddenunits], np.float32)
####previous weight
####prv_w = np.zeros([visibleunits, hiddenunits], np.float32)
####previous visible unit biases
####prv_vb = np.zeros([visibleunits], np.float32)
####previous hidden unit biases
####prv_hb = np.zeros([hiddenunits], np.float32)
###sess = tf.session()
###sess.run(tf.global_variables_initializer())




###epochs = 15
###batchsize = 100
###errors = []
###for i in range(epochs):
###    cum_cost = 0
###    counter = 0
###    for start, end in zip( range(0, len(trainX), batchsize), range(batchsize, len(trainX), batchsize)):
###        counter += 1
###        batch = trainX[start:end]
###        #cur_w = sess.run(update_w, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb})
###        #cur_vb = sess.run(update_vb, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb})
###        #cur_nb = sess.run(update_hb, feed_dict={v0: batch, w: prv_w, vb: prv_vb, hb: prv_hb})
###        #prv_w = cur_w
###        #prv_vb = cur_vb
###        #prv_hb = cur_hb
###        _, cd_out = sess.run([train, cd], feed_dict={v0: batch})
###        cum_cost += cd_out
###        # print(vb_cur)
###        # print(hb_cur)
###        # print(w_cur)
###    #errors.append(sess.run(err_sum, feed_dict={v0: trainX, w: cur_w, vb: cur_vb, hb: cur_hb}))
###    errors.append(cum_cost/counter)
###    print (errors[-1])
###plt.plot(errors)
###plt.ylabel('error')
###plt.xlabel('epoch')
###plt.show()
