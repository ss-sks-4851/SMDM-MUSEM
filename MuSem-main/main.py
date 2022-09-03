#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('train.csv')

df


# In[2]:


import os
#full path.
os.getcwd()


# In[3]:


files = os.listdir(os.getcwd())
files


# In[2]:


# df = pd.read_csv('train.csv')
# df=df.iloc[:999]
# df.to_csv("train1.csv",index=False)

# df = pd.read_csv('test.csv')
# df=df.iloc[:999]
# df.to_csv("test1.csv",index=False)


# In[3]:


print("Load csv files")

import csv
import json
import pickle

import re, unicodedata
import nltk
import inflect
from nltk import word_tokenize

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN

import numpy as np

from collections import Counter


# In[4]:


# df=df.iloc[:999]


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


data_directory = "./"



train_dataset_file_path = data_directory+"/train_dataset_300"
test_dataset_file_path = data_directory+"/test_dataset_300"

create_test_dataset = False


train_file_path = data_directory+"/train.csv"
test_file_path = data_directory+"/test.csv"


embedding_file_path = data_directory+"/glove.6B.300d.txt"
test_dataset_file_path = data_directory+"/test_dataset"


embedding_dim = 300
max_sen_len = 50

X_train = []
X_train_lenght = []
y_train = []

X_test = []
X_test_lenght = []


# In[8]:


with open(train_file_path, newline='',encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X_train.append([row[i] for i in [0, 1]])
        X_train_lenght.append([len(row[i]) for i in [0, 1]])
        y_train.append(row[2])
X_train = X_train[1:]
X_train_lenght = X_train_lenght[1:]
y_train = y_train[1:]
with open(test_file_path, newline='',encoding="utf8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        X_test.append([row[i] for i in [0, 1]])
        X_test_lenght.append([len(row[i]) for i in [0, 1]])
X_test = X_test[1:]
X_test_lenght = X_test_lenght[1:]


# In[9]:


y_train[:5]


# In[10]:


print("data preprocessing")

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def preprocessing(sample):
    words = nltk.word_tokenize(sample)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    return words

def load_embedding(embedding_file_path, wordset, embedding_dim):
    words_dict = dict()
    word_embedding = []
    index = 1
    words_dict['$EOF$'] = 0
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'r',encoding="utf-8") as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = np.array([float(s) for s in line[1:]])
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    return word_embedding, words_dict


# In[11]:


wordset = set()

i = 0

for line in X_train:
    print(i, "/", len(X_train))
    i +=1
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)

i = 0
for line in X_test:
    print(i, "/", len(X_test))
    i +=1
    line[0] = preprocessing(line[0])
    line[1] = preprocessing(line[1])
    for word in line[0]:
        wordset.add(word)
    for word in line[1]:
        wordset.add(word)

word_embedding, words_dict = load_embedding(embedding_file_path, wordset, embedding_dim)


# In[9]:


no_word_vector = np.zeros(embedding_dim)

for line in X_train:

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[0]) and line[0][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[0][i]]])
        else :
            sentence.append(no_word_vector)
    line[0] = np.array(sentence)

    sentence = []
    for i in range(max_sen_len):
        if i < len(line[1]) and line[1][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[1][i]]])
        else :
            sentence.append(no_word_vector)
    line[1] = np.array(sentence)


# In[10]:


argmax_y_train = []

for i in range(len(y_train)):
    if y_train[i] == '0':
        argmax_y_train.append(0)
    else :
        argmax_y_train.append(1)

for line in X_test:
    sentence = []
    for i in range(max_sen_len):
        if i < len(line[0]) and line[0][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[0][i]]])
        else :
            sentence.append(no_word_vector)
    line[0] = np.array(sentence)
    sentence = []
    for i in range(max_sen_len):
        if i < len(line[1]) and line[1][i] in words_dict:
            sentence.append(word_embedding[words_dict[line[1][i]]])
        else :
            sentence.append(no_word_vector)
    line[1] = np.array(sentence)

y_train = []
for i in range(len(argmax_y_train)):
    if argmax_y_train[i] == 0:
        y_train.append([1, 0])
    else :
        y_train.append([0, 1])


# In[11]:


# In[12]:


X_train[:5]


# In[13]:


y_train[:5]


# In[14]:


'''
DATASET RE-SAMPLING
'''
print("Original dataset shape ", Counter(argmax_y_train))


print("shape")
print(np.shape(X_train))
print(np.shape(y_train))


# In[15]:


'''
Split in train and test set
'''

test_percentage = 0.20

train_X = np.array(X_train[int(test_percentage*len(X_train)):])
train_X_lenght = np.array(X_train_lenght[int(test_percentage*len(X_train_lenght)):])
train_y = np.array(y_train[int(test_percentage*len(y_train)):])
test_X = np.array(X_train[:int(test_percentage*len(X_train))])
test_X_lenght = np.array(X_test_lenght[:int(test_percentage*len(X_test_lenght))])
test_y = np.array(y_train[:int(test_percentage*len(y_train))])

train_dataset = [train_X, train_X_lenght, train_y, test_X, test_X_lenght, test_y]


test_dataset = [X_test, X_test_lenght]


# In[16]:


print("shape")
print(np.shape(train_X))
print(np.shape(train_y))
print(np.shape(test_X))
print(np.shape(test_y))


# In[17]:


print("Original dataset shape ", Counter(argmax_y_train))


# In[12]:

'''
Save dataset
'''
print("Save dataset")

with open(train_dataset_file_path, 'wb') as f:
    pickle.dump(train_dataset, f, protocol=4)
print("train dataset done")

with open(test_dataset_file_path, 'wb') as f:
    pickle.dump(test_dataset, f, protocol=4)
print("test dataset done")


# In[19]:


import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, max_sen_len, class_num, embedding_dim, hidden_size):

        self.max_sen_len = max_sen_len
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.hidden_size = hidden_size

        with tf.name_scope('input'):
            self.x1 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="x1")
            self.x2 = tf.placeholder(tf.float32, [None, self.max_sen_len, self.embedding_dim], name="x2")
            self.y = tf.placeholder(tf.float32, [None, self.class_num], name="y")

        with tf.name_scope('weights'):
            self.weights = {
                'q_1_to_2': tf.Variable(tf.random_uniform([4*embedding_dim, self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size, 1], -0.01, 0.01)),

                'z': tf.Variable(tf.random_uniform([2*self.embedding_dim+self.hidden_size, self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.hidden_size, self.class_num], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'q_1_to_2': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'p_1_to_2': tf.Variable(tf.random_uniform([1], -0.01, 0.01)),

                'z': tf.Variable(tf.random_uniform([self.hidden_size], -0.01, 0.01)),

                'f': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
            }

    def inter_attention(self):

        x1_shape = tf.shape(self.x1)
        x2_shape = tf.shape(self.x2)

        x1_reshape = tf.reshape(self.x1, [-1, self.embedding_dim, 1])
        ones = tf.ones([x1_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x1_increase = tf.matmul(x1_reshape, ones)
        x1_increase = tf.transpose(x1_increase, perm=[0, 2, 1])
        x1_increase = tf.reshape(x1_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        x2_reshape = tf.reshape(self.x2, [-1, self.embedding_dim, 1])
        ones = tf.ones([x2_shape[0]*self.max_sen_len, 1, self.max_sen_len])
        x2_increase = tf.matmul(x2_reshape, ones)
        x2_increase = tf.transpose(x2_increase, perm=[0, 2, 1])
        x2_increase = tf.reshape(x2_increase, [-1, self.max_sen_len, self.max_sen_len, self.embedding_dim])
        x2_increase = tf.transpose(x2_increase, perm=[0, 2, 1, 3])
        x2_increase = tf.reshape(x2_increase, [-1, self.max_sen_len*self.max_sen_len, self.embedding_dim])

        concat = tf.concat([x1_increase, x2_increase], axis=-1)
        concat = tf.reshape(concat, [-1, 2*self.embedding_dim])

        dot = tf.multiply(x1_increase, x2_increase)
        dot = tf.reshape(dot, [-1, self.embedding_dim])

        substract = tf.math.subtract(x1_increase, x2_increase)
        substract = tf.reshape(substract, [-1, self.embedding_dim])

        s_1_to_2 = tf.nn.relu(tf.matmul(tf.concat([concat, dot, substract], axis=-1), self.weights['q_1_to_2']) + self.biases['q_1_to_2'])
        s_1_to_2 = tf.matmul(s_1_to_2, self.weights['p_1_to_2']) + self.biases['p_1_to_2']
        s_1_to_2 = tf.reshape(s_1_to_2, [-1, self.max_sen_len, self.max_sen_len])

        a_1 = tf.reshape(tf.nn.softmax(tf.reduce_max(s_1_to_2, axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_1_to_2 = tf.reshape(tf.matmul(a_1, self.x1), [-1, self.embedding_dim])

        a_2 = tf.reshape(tf.nn.softmax(tf.reduce_max(tf.transpose(s_1_to_2, perm=[0, 2, 1]), axis=-1), axis=-1), [-1, 1, self.max_sen_len])

        self.v_a_2_to_1 = tf.reshape(tf.matmul(a_2, self.x2), [-1, self.embedding_dim])

        self.v_a = tf.concat([self.v_a_1_to_2, self.v_a_2_to_1], axis=-1)

    def long_short_memory_encoder(self):

        lstm_cell = tf.keras.layers.LSTMCell(self.hidden_size)
        LSTM_layer = tf.keras.layers.RNN(lstm_cell)
        self.v_c = LSTM_layer(tf.concat([self.x1, self.x2], axis=1))

    def prediction(self):

        v = tf.concat([self.v_a, self.v_c], -1)
        v = tf.nn.relu(tf.matmul(v, self.weights['z']) + self.biases['z'])

        self.scores = tf.nn.softmax((tf.matmul(v, self.weights['f']) + self.biases['f']), axis=-1)

        self.predictions = tf.argmax(self.scores, -1, name="predictions")

    def build_model(self):

        self.inter_attention()
        self.long_short_memory_encoder()
        self.prediction()

        with tf.name_scope("loss"):

            losses = tf.nn.softmax_cross_entropy_with_logits(
                #labels=tf.argmax(self.y, -1),
                labels=self.y,
                logits=self.scores
            )

            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.y, -1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            self.c_matrix = tf.confusion_matrix(labels = tf.argmax(self.y, -1), predictions = self.predictions, name="c_matrix")



# In[20]:


import pickle
import os
import datetime
import time

import tensorflow as tf
import numpy as np


# In[15]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"

data_directory = "./"
backup_directory = "../Models/"

dataset_file_path = data_directory+"/train_dataset_300"

print("Restore Data")

with open(dataset_file_path, 'rb') as f:
    dataset = pickle.load(f)


# In[21]:


print("DATASET :", np.shape(dataset))
print("train_X :", np.shape(dataset[0]))
print("train_y :", np.shape(dataset[2]))
print("test_X :", np.shape(dataset[3]))
print("test_y :", np.shape(dataset[5]))


# In[17]:


train_X = np.array(dataset[0])
#train_X_lenght = np.array(dataset[1])
train_y = np.array(dataset[2])
test_X = np.array(dataset[3])
#test_X_lenght = np.array(dataset[4])
test_y = np.array(dataset[5])


# In[21]:


n_class = 2
embedding_dim = 300
max_sen_len = 50

hidden_size = 100

learning_rate = 0.001
batch_size = 100
test_batch_size = 200
num_epochs = 10
#evaluate_every = 500
evaluate_every = 500

nb_batch_per_epoch = int(len(train_X)/batch_size+1)
nb_batch_per_epoch_test = int(len(test_X)/test_batch_size+1)

allow_soft_placement = True
log_device_placement = False


# In[22]:


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[23]:


with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=allow_soft_placement,
        log_device_placement=log_device_placement
    )
    session_config.gpu_options.allow_growth = False
    session_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=session_config)

    with sess.as_default():

        print("Model creation")

        model = Model(
        max_sen_len = max_sen_len,
        embedding_dim = embedding_dim,
        class_num = n_class,
        hidden_size = hidden_size
        )

        print("Model construction")

        model.build_model()

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(model.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath(backup_directory+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model_dpc")

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        best_accuracy = 0.
        predict_round = 0

        for epoch in range(num_epochs):
            indices = np.arange(len(train_X))
            np.random.shuffle(indices)
            train_X = train_X[indices]
            train_y = train_y[indices]

            print(epoch)
            for batch in range(nb_batch_per_epoch-1):

                idx_min = batch * batch_size
                idx_max = min((batch+1) * batch_size, len(train_X)-1)
                x1 = train_X[idx_min:idx_max, 0]
                x2 = train_X[idx_min:idx_max, 1]
                y = train_y[idx_min:idx_max]
                # normalized batch :

                print(x1.shape)
                print(x2.shape)
                print(y.shape)
                feed_dict = {
                    model.x1: x1,
                    model.x2: x2,
                    model.y: y,
                    #model.class_weights: class_weights,
                    #model.class_weights_accuracy: class_weights_accuracy,
                }

                _, step, loss, accuracy, c_matrix = sess.run(
                    [train_op, global_step, model.loss, model.accuracy, model.c_matrix],
                    feed_dict=feed_dict)

                time_str = datetime.datetime.now().isoformat()
                f1_scr = (2*c_matrix[1][1])/((2*c_matrix[1][1])+c_matrix[1][0]+c_matrix[0][1])
                print("{}: step {}/{}, loss {:g}, acc {:g}, F1 Score {:g}".format(time_str, step, num_epochs*nb_batch_per_epoch, loss, accuracy,f1_scr))
                #print(c_matrix[0][1])
                #print(c_matrix)

                current_step = tf.train.global_step(sess, global_step)
                print(current_step)
                if current_step % evaluate_every == 0:
                    predict_round += 1
                    print("\nEvaluation round %d:" % (predict_round))

                    indices = np.arange(len(test_X))
                    np.random.shuffle(indices)
                    test_X = test_X[indices]
                    test_y = test_y[indices]

                    accuracy = 0
                    c_matrix = np.zeros((n_class, n_class))

                    for test_batch in range(nb_batch_per_epoch_test):
                        idx_min = test_batch * test_batch_size
                        idx_max = min((test_batch+1) * test_batch_size, len(test_X)-1)

                        x1 = test_X[idx_min:idx_max, 0]
                        x2 = test_X[idx_min:idx_max, 1]

                        y = test_y[idx_min:idx_max]



                        feed_dict = {
                            model.x1: x1,
                            model.x2: x2,
                            model.y: y,
                            #model.class_weights: class_weights,
                            #model.class_weights_accuracy: class_weights_accuracy,
                        }

                        batch_accuracy, batch_c_matrix = sess.run([model.accuracy, model.c_matrix], feed_dict=feed_dict)
                        accuracy = accuracy + batch_accuracy
                        c_matrix = np.add(c_matrix, batch_c_matrix)

                    accuracy = accuracy/nb_batch_per_epoch_test
                    print("Test acc {:g}".format(accuracy))
                    print("C_matrix ", c_matrix)

                    if accuracy >= best_accuracy:
                        best_accuracy = accuracy
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))


# In[ ]:
