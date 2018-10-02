#
#三层全连接神经网络
#
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import os
import numpy as np
import tqdm
#神经网络中一些超参数
import warnings
warnings.filterwarnings('ignore')


#构建简单神经网络模型


if __name__ == "__main__":
    # pandas读取json数据
    print('读取json数据...')
    ##df_train(39774, 3) ['cuisine', 'id', 'ingredients']
    df_train = pd.read_json('all/train.json')
    ##df_test(9944, 2) ['cuisine', 'id', 'ingredients']
    df_test = pd.read_json('all/test.json')
    # 整理数据
    ##将字符型离散特征提取出来，使用TF-IDF编码
    print('TF-IDF编码...')
    ing_train = df_train.ingredients
    ing_test = df_test.ingredients
    text_ing_train = [' '.join(x) for x in ing_train]
    text_ing_test = [' '.join(x) for x in ing_test]
    tdidf = TfidfVectorizer(binary=True)
    td_fit = tdidf.fit(text_ing_train)
    ###X_train(39774, 3010)
    ###X_test(9944, 3010)
    X_train = td_fit.transform(text_ing_train)
    X_test = td_fit.transform(text_ing_test)
    ##得到标签，将标签转化为数值型，使用LabelEncoder
    #print('LabelEncoder编码...')
    Y_train_no = df_train.cuisine
    #Y_train_no['greek', 'southern_us', 'filipino', 'indian', 'indian', 'jamaican', 'spanish', ...]
    Y_train_no = Y_train_no.tolist()

    vectorizer = CountVectorizer(min_df=1)
    #Y_train:[0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0]
    Y_train = vectorizer.fit_transform(Y_train_no).toarray()
    #print(type(Y_train),Y_train[0])
    #lb = LabelEncoder()
    #Y_train = lb.fit_transform(Y_train)
    #还需要将Y_train做成one-hot的形式，最后尺寸为(33807,24)
    Y_train = pd.DataFrame(Y_train)
    ##切分出一部分当作验证集
    print('切分验证集...')
    ###X_train(33807, 3010) X_val(5967,3010) Y_train(33807,)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.85)
    #print(type(X_train), type(X_val), type(Y_train), type(Y_val))

    ###数据类型从<class 'scipy.sparse.csr.csr_matrix'>转化为<class 'numpy.ndarray'>,Y本来就是ndarray
    X_train, X_val  = X_train.A, X_val.A

    #Y_train = Y_train.reshape(-1,1)
    X = tf.placeholder(dtype=tf.float32,shape=[None,3010],name='X')
    Y = tf.placeholder(dtype=tf.float32,shape=[None,20],name='Y')
    print('开始训练...')
    dense1 = tf.layers.dense(inputs=X,units=1024,activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense2,units=20,activation=None)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    train_op = tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(50):
            avg_cost = 0
            total_batch = int(len(X_train)/256)
            for i in range(total_batch):
                batch_x, batch_y = X_train[i*256:(i+1)*256][:], Y_train[i*256:(i+1)*256][:]
                _,c = sess.run([train_op,cost],feed_dict={X:batch_x, Y:batch_y})
                avg_cost += c/total_batch
            '''
            batch_x, batch_y = X_train[total_batch*256:][:]
            _,c = sess.run([train_op,cost],feed_dict={X_train:batch_x, Y_train:batch_y})
            avg_cost += c
            avg_cost = avg_cost/total_batch
            '''
            if (epoch+1) % 10 == 0:
                print("epoch.{}.cost={}".format(epoch+1, avg_cost))
                acc = sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1)), tf.float32)),feed_dict={X:X_val, Y:Y_val})
                print("acc in epoch{} is {}".format(epoch+1, acc))
    print('end')
