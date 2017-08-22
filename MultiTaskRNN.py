
# coding: utf-8

# In[1]:

from __future__ import print_function

#standard libraries
import csv
import os
import time
import re
import struct
import string
import zipfile
import random
from functools import reduce

#custom libraries
import tensorflow as tf
import pandas as pd
import numpy as np
from IPython.display import clear_output


# ### Run this on server and restart jupyter notebook

# In[2]:

# export LD_LIBRARY_PATH=../../../../../usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH


# In[3]:

def process_example(example, args):
    example = "{:<{}}".format(example, args.max_example_string_len)
    example = example.lower()
    example = list(example)
    example = example[:args.max_example_string_len]
    example = [i for i in example]
    example = [args.char_vocab.get(x) if args.char_vocab.get(x) is not None                else args.char_vocab.get(b" ") for x in example]
    example = np.array(example)
    example = example.astype(np.int64)
    return example


# In[4]:

def process_label1(task_1, args):
    task_1 = task_1.lower()
    task_1 = "".join(task_1.split())
    label1 = np.array([args.task_1_vocab.get(task_1)])
    return label1


# In[5]:

def process_label2(label2, args):
    label2 = re.sub("by ", "", label2)
    label2 = label2.lower()
    label2 = "".join(label2.split())
    if args.task_2_vocab.get(label2) is not None:
        label2 = np.array([args.task_2_vocab.get(label2)])
    else:            
        label2 = np.array([args.task_2_vocab_len]) 
    return label2


# In[6]:

def process_aux_label(example, label2, args):
    example = "{:<{}}".format(example, args.max_example_string_len)
    example = example.lower()
    example = example.split()
    
    label2 = str(label2)
    label2 = re.sub("by ", "", label2)
    label2 = label2.lower()
    label2 = label2.split()
    label2 = set(label2)

    aux_label = ["0" * len(example[i]) if example[i] not in label2 else "1" * len(example[i]) for i in range(len(example))]
    aux_label = "0".join(aux_label)
    aux_label = "{:<{}}".format(aux_label, args.max_example_string_len)
    aux_label = aux_label.replace(" ", "0")
    aux_label = list(aux_label)
    aux_label = np.array([int(x) for x in aux_label])
    return aux_label


# In[7]:

def read_file_format(filename_queue, args):
    if args.evaluation_time:
        reader = tf.TextLineReader(skip_header_lines=1)
    else:
        reader = tf.TextLineReader(skip_header_lines=args.skip_example_rows + 1)
    _, value = reader.read(filename_queue)

    record_defaults =  [
        tf.constant(["NA"], dtype=tf.string),
        tf.constant(["NA"], dtype=tf.string),
        tf.constant(["NA"], dtype=tf.string)
    ]
    
    label_1, label_2, example = tf.decode_csv(
        value, record_defaults=record_defaults, field_delim=","
    )
    
    if args.character_level_training:
        example = tf.cast(example, tf.string)
        example = tf.py_func(lambda x: process_example(x, args=args), [example], tf.int64)
        example = tf.one_hot(example, depth=args.char_vocab_len)
        example = tf.reshape(example, [args.max_example_string_len, args.char_vocab_len])
    
    else:
        raise Exception("Needs fixing y'all")
#         example = example
#         example = tf.py_func(lambda x: x.decode("ISO-8859-1"), [example], tf.string)
#         example = tf.py_func(lambda x: args.example_vocab.get(x), [example], tf.string)
#         example = tf.string_to_number(example, tf.int32)
#         example = tf.one_hot(example, depth=args.example_vocab_len)

#         if args.sentence_rep == "bow":
#             example = tf.reduce_sum(example, -2)
#             example = tf.reshape(example, [args.example_vocab_len])
        

    label1 = tf.py_func(lambda x: process_label1(x, args=args),
                        [label_1], tf.int64)
    label1 = tf.one_hot(label1, depth=args.task_1_vocab_len)
    label1 = tf.reshape(label1, [args.task_1_vocab_len])

    label2 = tf.py_func(lambda x: process_label2(x, args=args),
                        [label_2], tf.int64)
    label2 = tf.one_hot(label2, depth=args.task_2_vocab_len)
    label2 = tf.reshape(label2, [args.task_2_vocab_len])
    
    aux_label = tf.py_func(lambda x, y: process_aux_label(x, y, args=args),
                           [example, task_2], tf.int64)
    aux_label = tf.reshape(aux_label, [args.max_example_string_len])

    
    return example, label1, label2, aux_label


# In[8]:

def input_pipeline(filenames, args):   
    filename_queue = tf.train.string_input_producer(
        filenames, num_epochs=args.num_epochs, shuffle=False)

    example, label1, label2, aux_label = read_file_format(filename_queue=filename_queue,
                                      args=args)
        
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * args.batch_size
    
    if args.character_level_training:
        batch_size = args.batch_size
        enqueued_tensors = [example, label1, label2, aux_label]
        if args.shuffle_batches:
            raise Exception("args.shuffle_batches must be set to False if args.character_level_training             is True. You don't want to shuffle your character batches.")
        else:
            example_batch, label1_batch, label2_batch, aux_label_batch = tf.train.batch(
                tensors=enqueued_tensors, batch_size=batch_size, capacity=capacity)

            example_batch = tf.reshape(example_batch, (batch_size, args.max_example_string_len, args.char_vocab_len))
        
    else:
        batch_size = args.batch_size
        enqueued_tensors = [example, label1, label2, aux_label]
        if args.shuffle_batches:
            example_batch, label1_batch, label2_batch, aux_label_batch = tf.train.shuffle_batch(
                tensors=enqueued_tensors, batch_size=batch_size, capacity=capacity, 
                min_after_dequeue=min_after_dequeue)   
        else:
            example_batch, label1_batch, label2_batch, aux_label_batch = tf.train.batch(
                tensors=enqueued_tensors, batch_size=batch_size, capacity=capacity)

    return example_batch, label1_batch, label2_batch, aux_label_batch


# In[9]:

def _add_loss_summaries(total_loss, averager=None, include_averaged_loss=False):
    losses = tf.get_collection('losses')
    if include_averaged_loss:
        loss_averages_op = averager.apply(losses + [total_loss])

    for l in losses + [total_loss]:

        l_name = l.name.replace(":", "_")

        tf.summary.scalar(l_name + '_raw_', tf.reduce_sum(l))        
        if include_averaged_loss:
            tf.summary.scalar(l_name + '_raw_', l)
            tf.summary.scalar(l_name, averager.average(l))
        
    if include_averaged_loss:
        return loss_averages_op
    else:
        return total_loss


# In[10]:

class LSTM_Cell:
    
    def __init__(self, args, input_size=None, output_size=None, is_aux_cell=False, scope_name=None, current_weights=None):

        if input_size is None:
            if args.character_level_training:
                if args.use_convolution:
                    self.input_size = args.batch_size + args.conv_out_size 
            else:
                if args.sentence_rep == "bow":
                    self.input_size = args.batch_size
                else:
                    self.input_size = args.batch_size * args.conv_out_size
        else:
            self.input_size = input_size
            
        if output_size is None:
            self.rnn_size = args.rnn_size
        else:
            self.rnn_size = output_size
            
        self.state_size = self.rnn_size * 2
        
        with tf.variable_scope(scope_name, reuse=None):
            if args.weight_noise_type == None:
                self.W = tf.get_variable('W', [self.input_size + self.rnn_size, 4 * self.rnn_size],
                                         tf.float32, tf.random_normal_initializer(), trainable=True)

            elif args.weight_noise_type == "static":
                self.W = tf.get_variable('W', [self.input_size + self.rnn_size, 4 * self.rnn_size],
                                         tf.float32, tf.random_normal_initializer(), trainable=True)
                weight_noise = tf.truncated_normal([self.input_size + self.rnn_size, 4 * self.rnn_size],
                                                   stddev=args.weight_prior_variance)
                self.W = self.W + weight_noise        

            elif args.weight_noise_type == "adaptive":
                self.W = tf.reshape(current_weights,
                                    [(self.input_size + self.rnn_size),  4 * self.rnn_size])
        
            self.b = tf.get_variable('b', [1, self.rnn_size * 4], tf.float32,
                                     tf.constant_initializer(0.0), trainable=True)
            
    def __call__(self, i, state):
        self.state = tf.reshape(state, [args.batch_size, self.rnn_size * 2])
        self.c_prev = tf.slice(self.state, [0, 0], [-1, self.rnn_size])
        self.h_prev = tf.slice(self.state, [0, self.rnn_size], [-1, self.rnn_size])
        
        i = tf.reshape(i, [args.batch_size, self.input_size])

        
        data = tf.concat([i, self.h_prev], 1)
        data = tf.reshape(data, [args.batch_size, (self.input_size + self.rnn_size)])

        
        weighted = tf.matmul(data, self.W)

        self.i, self.j, self.f, self.o = tf.split(weighted, num_or_size_splits=4, axis=1)
        self.i_b, self.j_b, _, self.o_b = tf.split(self.b, num_or_size_splits=4, axis=1)
        
        old = tf.sigmoid(self.f + args.forget_bias) * self.c_prev
        new =  tf.sigmoid(self.i + self.i_b) * tf.tanh(self.j + self.j_b)
        self.c = (old + new)
        self.h = tf.sigmoid(self.o + self.o_b) * tf.tanh(self.c)
        
        self.state = tf.concat([self.c, self.h], axis=1)
        
        self.h = tf.reshape(self.h, [args.batch_size, self.rnn_size])
        self.state = tf.reshape(self.state, [args.batch_size, self.state_size])

        return self.h, self.state
    
    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.state_size], dtype=dtype)


# In[11]:

class ReLU_Cell:
    
    def __init__(self, args, input_size=None, scope_name=None, current_weights=None):
        self.relu_size = args.relu_size
        if input_size is None:
            if args.character_level_training:
                if args.use_convolution:
                    self.input_size = args.batch_size + args.conv_out_size 
            else:
                if args.sentence_rep == "bow":
                    self.input_size = args.batch_size
                else:
                    self.input_size = args.batch_size * args.conv_out_size
        else:
            self.input_size = input_size
        
        with tf.variable_scope(scope_name, reuse=None):
            if args.weight_noise_type == None:
                self.W = tf.get_variable('W', [self.input_size, self.relu_size],
                                         tf.float32, tf.random_normal_initializer(), trainable=True)

            elif args.weight_noise_type == "static":
                self.W = tf.get_variable('W', [self.input_size, self.relu_size],
                                         tf.float32, tf.random_normal_initializer(), trainable=True)
                weight_noise = tf.truncated_normal([self.input_size, self.relu_size],
                                                   stddev=args.weight_prior_variance)
                self.W = self.W + weight_noise        

            elif args.weight_noise_type == "adaptive":
                self.W = tf.reshape(current_weights, [self.input_size, self.relu_size])

            self.b = tf.get_variable('b', [1, self.relu_size], tf.float32,
                                     tf.constant_initializer(0.0), trainable=True)
        
            
    def __call__(self, i, state):
        i = tf.reshape(i, [args.batch_size, self.input_size])
        
        self.activation = tf.add(tf.matmul(i, self.W), self.b)
        self.activation = tf.maximum(tf.zeros(self.activation.shape), self.activation)

        self.state = tf.eye(self.relu_size, batch_shape=[args.batch_size])
        
        self.activation = tf.reshape(self.activation, [args.batch_size, self.relu_size])
        return self.activation, self.state
    
    def zero_state(self, batch_size, dtype):
        return tf.zeros([batch_size, self.relu_size], dtype=dtype)


# In[12]:

def log_likelihood(y1=None, y2=None, aux_y=None, aux_logit=None, logit1=None, logit2=None, args=None):
    nll1 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=logit1), axis=None)
    nll2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y2, logits=logit2), axis=None)

    if args.use_aux_task:
        aux_nll = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=aux_y, logits=aux_logit), axis=None)
        return -tf.reduce_sum([nll1, nll2, aux_nll], axis=None)
    else:
        return -tf.reduce_sum([nll1, nll2], axis=None)


# In[13]:

def evaluate_label1_accuracy(y_hat, y_):
    correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    return accuracy


# In[14]:

def evaluate_label2_accuracy(y_hat, y_):
    correct = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y_, 1))
    tf.summary.scalar('validation_accuracy', accuracy)
    return accuracy


# In[15]:

class Model():
    
    def __init__(self, args):
        print("Initializing input variables.")
        self.batch_size = args.batch_size
        
        with tf.device("/cpu:0"):
            if args.character_level_training:
                self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.max_example_string_len, args.char_vocab_len])
                self.y1_ = tf.placeholder(tf.float32, shape=[args.batch_size, args.task_1_vocab_len])
                self.y2_ = tf.placeholder(tf.float32, shape=[args.batch_size, args.task_2_vocab_len])
                if args.use_aux_task:
                    self.aux_y = tf.placeholder(tf.float32, shape=[args.batch_size, args.max_example_string_len])

            else:
                if args.sentence_rep=='bow':
                    self.x = tf.placeholder(tf.float32, shape=[args.batch_size, args.example_vocab_len])
                    self.y1_ = tf.placeholder(tf.float32, shape=[args.batch_size, args.task_1_vocab_len])
                    self.y2_ = tf.placeholder(tf.float32, shape=[args.batch_size, args.task_2_vocab_len])

                else:
                    raise Exception("Only Bag-o-Words feature representation available;                     require args.sentence_rep=='bow'")


            if args.weight_noise_type == "adaptive":
                print("Initializing weights with adaptive noise.")

                task_1_sfmx_weight_size = (args.relu_size * args.task_1_vocab_len)
                task_2_sfmx_weight_size = (args.relu_size * args.task_2_vocab_len)

                if args.character_level_training:
                    if args.use_convolution:
                        if args.conv_padding.lower()=="same": 
                            if args.use_embeddings:
                                args.conv_out_size = (args.max_example_string_len * args.embedding_size)  // (args.conv_stride)
                            else:
                                args.conv_out_size = (args.max_example_string_len)  // (args.conv_stride)

                        elif args.conv_padding.lower()=="valid":    
                            if args.use_embeddings:
                                args.conv_out_size = ((args.max_example_string_len - (args.conv_width - 1))                                                       * args.embedding_size)  // (args.conv_stride)
                            else:
                                args.conv_out_size = (args.max_example_string_len - (args.conv_width - 1)) // (args.conv_stride)

                        else:
                            raise Exception("args.conv_padding=={} not implemented".format(args.conv_padding))

                        if args.lateral_connections:
                            task_1_weight_size_1 = (args.rnn_size) * args.relu_size
                            task_1_weight_size = task_1_weight_size_1 + args.relu_size *                                                     args.relu_size * 2 * (args.task_1_layers-1)

                            task_2_weight_size_1 = (args.rnn_size) * args.relu_size
                            task_2_weight_size = task_2_weight_size_1 + args.relu_size *                                                     args.relu_size * 2 * (args.task_2_layers-1) 
                        else:
                            task_1_weight_size_1 = (args.rnn_size) * args.relu_size
                            task_1_weight_size = task_1_weight_size_1 + args.relu_size *                                                     args.relu_size * (args.task_1_layers-1)

                            task_2_weight_size_1 = (args.rnn_size) * args.relu_size
                            task_2_weight_size = task_2_weight_size_1 + args.relu_size *                                                     args.relu_size * (args.task_2_layers-1) 

                        if args.use_embeddings:

                            embedding_weight_size = args.char_vocab_len * args.embedding_size 

                            conv_size = args.conv_width * args.embedding_size * args.conv_out_channels

                            shared_weight_size_1 = (args.rnn_size + args.conv_out_size * args.conv_out_channels) *                                                 (4 * args.rnn_size) 

                            shared_weight_size = shared_weight_size_1 + ((args.rnn_size + args.rnn_size) *                                             (4 * args.rnn_size) * (args.shared_layers - 1))

                            weight_size = conv_size + shared_weight_size + task_1_weight_size + task_2_weight_size                                             + task_1_sfmx_weight_size + task_2_sfmx_weight_size + embedding_weight_size

                        else:
                            conv_size = args.conv_width * args.char_vocab_len * args.conv_out_channels

                            shared_weight_size_1 = (args.rnn_size + args.conv_out_size * args.conv_out_channels) *                                                 (4 * args.rnn_size) 

                            shared_weight_size = shared_weight_size_1 + ((args.rnn_size + args.rnn_size) *                                             (4 * args.rnn_size) * (args.shared_layers - 1))
                            
                            weight_size = conv_size + shared_weight_size + task_1_weight_size                                             + task_2_weight_size + task_1_sfmx_weight_size + task_2_sfmx_weight_size

                        if args.use_aux_task:
                            attention_size_1 = ((args.aux_rnn_size + args.conv_out_size * args.conv_out_channels) +                                                 args.aux_rnn_size) * (4 * args.aux_rnn_size)
                            attention_size = attention_size_1 + ((args.aux_rnn_size + args.aux_rnn_size) *                                                 (4 * args.aux_rnn_size) * (args.aux_layers - 1))
                            attention_out_size = args.aux_rnn_size * args.max_example_string_len
                            weight_size = weight_size + attention_size + attention_out_size
                            
                    else:
                        raise Exception("Nah too big fam.")

                else:
                    if args.sentence_rep == "bow":
                        print("Using bag-o-words representation.")
                        shared_weight_size = (args.batch_size + args.example_vocab_len) *                                         (4 * args.rnn_size) * args.shared_layers

                    elif args.sentence_rep == "conv":
                        print("Using sequential representation.")
                        conv_size = args.batch_size * args.conv_width * args.example_vocab
                        shared_weight_size = conv_size 

                    if lateral_connections:
                        task_1_weight_size = (args.batch_size + args.example_vocab_len) * args.relu_size * 2 * args.task_1_layers
                        task_2_weight_size = (args.batch_size + args.example_vocab_len) * args.relu_size * 2 * args.task_2_layers
                    else:
                        task_1_weight_size = (args.batch_size + args.example_vocab_len) * args.relu_size * args.task_1_layers
                        task_2_weight_size = (args.batch_size + args.example_vocab_len) * args.relu_size * args.task_2_layers

                print("Initializing weight priors. {0}-dimensional prior needed.".format(weight_size))
                prior_loc = [0.] * weight_size
                prior_scale_diag = [args.weight_prior_variance] * weight_size
                self.weight_prior = tf.contrib.distributions.Normal(
                        prior_loc,
                        prior_scale_diag)
            
                with tf.variable_scope("weights", reuse=None):
                    S_hat = tf.get_variable("S_hat",
                                            initializer=prior_scale_diag, trainable=True)
                    S = tf.exp(S_hat)
                    mu = tf.get_variable("mu",
                                         initializer=prior_loc, trainable=True)

                print("Initializing variational distributions for weights.")
                self.weight_dist = tf.contrib.distributions.Normal(mu, S)

                self.weight_st = tf.contrib.bayesflow.stochastic_tensor.StochasticTensor(self.weight_dist)

                current_weights = tf.squeeze(self.weight_st, name="W")

                if args.use_convolution:
                    if args.use_embeddings:
                        print("Initializing with embedding matrix; {0} weights required".format(embedding_weight_size))
                        if args.use_aux_task:
                            conv_W, shared_W, task_1_W, task_2_W, c_sfmx_W, b_sfmx_W, embedding_mat, attention_W, attention_out_W = tf.split(
                                    current_weights,
                                    num_or_size_splits=[conv_size,
                                                        shared_weight_size,
                                                        task_1_weight_size,
                                                        task_2_weight_size,
                                                        task_1_sfmx_weight_size,
                                                        task_2_sfmx_weight_size,
                                                        embedding_weight_size,
                                                        attention_size,
                                                        attention_out_size])
                        else:
                            conv_W, shared_W, task_1_W, task_2_W, c_sfmx_W, b_sfmx_W, embedding_mat = tf.split(
                                    current_weights,
                                    num_or_size_splits=[conv_size,
                                                        shared_weight_size,
                                                        task_1_weight_size,
                                                        task_2_weight_size,
                                                        task_1_sfmx_weight_size,
                                                        task_2_sfmx_weight_size,
                                                        embedding_weight_size])
                    else:
                        if args.use_aux_task:
                            conv_W, shared_W, task_1_W, task_2_W, c_sfmx_W, b_sfmx_W, attention_W, attention_out_W = tf.split(
                                current_weights,
                                num_or_size_splits=[conv_size,
                                                    shared_weight_size,
                                                    task_1_weight_size,
                                                    task_2_weight_size,
                                                    task_1_sfmx_weight_size,
                                                    task_2_sfmx_weight_size,
                                                    attention_size,
                                                    attention_out_size])
                        else:
                            conv_W, shared_W, task_1_W, task_2_W, c_sfmx_W, b_sfmx_W = tf.split(
                                current_weights,
                                num_or_size_splits=[conv_size,
                                                    shared_weight_size,
                                                    task_1_weight_size,
                                                    task_2_weight_size,
                                                    task_1_sfmx_weight_size,
                                                    task_2_sfmx_weight_size])

                print("Splitting up weight tensors.")
                shared_W1 = tf.split(shared_W, num_or_size_splits=[shared_weight_size_1,
                                                                   shared_weight_size - shared_weight_size_1])
                if args.shared_layers > 1:
                    shared_W = tf.split(shared_W1[1], num_or_size_splits=args.shared_layers-1)

                task_1_W1 = tf.split(task_1_W, num_or_size_splits=[task_1_weight_size_1,
                                                                       task_1_weight_size - task_1_weight_size_1])
                if args.task_1_layers > 1:
                    task_1_W = tf.split(task_1_W1[1], num_or_size_splits=args.task_1_layers-1)

                task_2_W1 = tf.split(task_2_W, num_or_size_splits=[task_2_weight_size_1,
                                                                 task_2_weight_size - task_2_weight_size_1])
                if args.task_2_layers > 1:
                    task_2_W = tf.split(task_2_W1[1], num_or_size_splits=args.task_2_layers-1)
                    
                attention_W1 = tf.split(attention_W, num_or_size_splits=[attention_size_1,
                                                                 attention_size - attention_size_1])
                if args.aux_layers > 1:
                    attention_W = tf.split(attention_W1[1], num_or_size_splits=args.aux_layers-1)
                print("Weight initialization complete.")

            else:
                shared_W = [None for i in range(args.shared_layers)]

            if args.weight_noise_type in (None, 'static'):
                if args.conv_padding.lower()=="same": 
                    if args.use_embeddings:
                        args.conv_out_size = (args.max_example_string_len * args.embedding_size)  // (args.conv_stride)
                    else:
                        args.conv_out_size = (args.max_example_string_len)  // (args.conv_stride)

                elif args.conv_padding.lower()=="valid":    
                    if args.use_embeddings:
                        args.conv_out_size = ((args.max_example_string_len - (args.conv_width - 1))                                               * args.embedding_size)  // (args.conv_stride)
                    else:
                        args.conv_out_size = (args.max_example_string_len - (args.conv_width - 1)) // (args.conv_stride)
                
                with tf.variable_scope("weights", reuse=None):
                    if args.use_aux_task:
                        self.attention_out_W = tf.get_variable('attention_W', [args.aux_rnn_size, args.max_example_string_len],
                                                           tf.float32, tf.random_normal_initializer, trainable=True)
                    self.c_sfmx_W = tf.get_variable('c_sfmx_W', [args.relu_size, args.task_1_vocab_len],
                                                tf.float32, tf.random_normal_initializer(), trainable=True)
                    self.b_sfmx_W = tf.get_variable('b_sfmx_W', [args.relu_size, args.task_2_vocab_len],
                                tf.float32, tf.random_normal_initializer(), trainable=True)
                    
                    if args.weight_noise_type == 'static':
                        print("Initializing weights with static noise.")
                        if args.use_aux_task:
                            attention_out_noise = tf.truncated_normal([args.aux_rnn_size, args.max_example_string_len],
                                                            stddev=args.weight_prior_variance)
                            self.attention_out_W = self.attention_out_W + attention_out_noise

                        c_sfmx_noise = tf.truncated_normal([args.relu_size, args.task_1_vocab_len],
                                                        stddev=args.weight_prior_variance)
                        b_sfmx_noise = tf.truncated_normal([args.relu_size, args.task_2_vocab_len],
                                            stddev=args.weight_prior_variance)
                        self.c_sfmx_W = self.c_sfmx_W + c_sfmx_noise
                        self.b_sfmx_W = self.b_sfmx_W + b_sfmx_noise

                    if args.character_level_training:
                        if args.use_embeddings:
                            self.embedding_mat = tf.get_variable('embedding', [args.embedding_size, args.char_vocab_len],
                                                        tf.float32, tf.random_normal_initializer(), trainable=True)
                            if args.weight_noise_type == 'static':
                                embedding_noise = tf.truncated_normal([args.embedding_size, args.char_vocab_len],
                                                                  stddev=args.weight_prior_variance)
                                self.embedding_mat = embedding_mat + embedding_noise
                            
                            if args.use_convolution:
                                self.conv_W = tf.get_variable('conv_W',
                                                              [args.conv_width, args.embedding_size, 1, args.conv_out_channels],
                                                              tf.float32,
                                                              tf.random_normal_initializer,
                                                              trainable=True)
                                if args.weight_noise_type == 'static':
                                    conv_noise = tf.truncated_normal([args.conv_width, args.embedding_size, 1, args.conv_out_channels],
                                                                  stddev=args.weight_prior_variance)
                                    self.conv_W = self.conv_W + conv_noise
                        else:
                            if args.use_convolution:
                                self.conv_W = tf.get_variable('conv_W',
                                                              [args.conv_width, args.char_vocab_len, args.conv_out_channels],
                                                              tf.float32,
                                                              tf.random_normal_initializer,
                                                              trainable=True)
                                if args.weight_noise_type == 'static':
                                    conv_noise = tf.truncated_normal(
                                        [args.conv_width, args.char_vocab_len, args.conv_out_channels],
                                        stddev=args.weight_prior_variance)
                                    self.conv_W = self.conv_W + conv_noise

            elif args.weight_noise_type == "adaptive":
                if args.character_level_training:
                    if args.use_convolution:
                        if args.use_embeddings:
                            self.conv_W = tf.reshape(conv_W, [args.conv_width, args.embedding_size, 1, args.conv_out_channels])
                        else:
                            self.conv_W = tf.reshape(conv_W, [args.conv_width, args.char_vocab_len, args.conv_out_channels])
                    if args.use_embeddings:
                        self.embedding_mat = tf.reshape(embedding_mat, [args.embedding_size, args.char_vocab_len])

                if args.use_aux_task:
                    self.attention_out_W = tf.reshape(attention_out_W, [args.aux_rnn_size, args.max_example_string_len])
                self.c_sfmx_W = tf.reshape(c_sfmx_W, [args.relu_size, args.task_1_vocab_len])
                self.b_sfmx_W = tf.reshape(b_sfmx_W, [args.relu_size, args.task_2_vocab_len])


            else:
                raise Exception("Unrecognized value for weight_noise_type; " +
                                "recognized values are: None, 'static', and 'adaptive'.")
            
            with tf.variable_scope("weights", reuse=None):
                self.task_1_b = tf.get_variable('task_1_b', [1, args.task_1_vocab_len], tf.float32,
                                    tf.constant_initializer(0.0), trainable=True)
                self.task_2_b = tf.get_variable('task_2_b', [1, args.task_2_vocab_len], tf.float32,
                        tf.constant_initializer(0.0), trainable=True)
                if args.use_aux_task:
                    self.aux_b = tf.get_variable('attention_b', [1, args.max_example_string_len], tf.float32,
                                                 tf.constant_initializer(1.0), trainable=True)
                
                if args.weight_noise_type == "adaptive":
                    print("Initializing LSTM cells; {0} weights required".format(shared_weight_size))
                else:
                    shared_W1 = [None]
                    shared_W = [None for i in range(args.shared_layers)]
                
                self.shared_lstm_cells = []
                for i in range(args.shared_layers):
                    if i==0:
                        if args.use_convolution:
                            temp_cell = LSTM_Cell(args,
                                                  input_size=args.conv_out_size * args.conv_out_channels,
                                                  scope_name="shared_lstm_{0}".format(i),
                                                  current_weights=shared_W1[0])
                        else:
                            raise Exception("boi u lost.")
                    else:
                        temp_cell = LSTM_Cell(args,
                                              input_size=args.rnn_size,
                                              scope_name="shared_lstm_{0}".format(i),
                                              current_weights=shared_W[i-1])
                    self.shared_lstm_cells.append(temp_cell)

                self.shared_cells = tf.contrib.rnn.MultiRNNCell(self.shared_lstm_cells)
                self.shared_initial_state = self.shared_cells.zero_state(args.batch_size, tf.float32)
                
                if args.use_aux_task:
                    if args.weight_noise_type == "adaptive":
                        print("Initializing attention cells; {0} weights required.".format(attention_size))
                    else:
                        attention_W1 = [None]
                        attention_W = [None for i in range(args.aux_layers)]

                    self.attention_cell_list = []
                    for i in range(args.aux_layers):
                        if i==0:
                            temp_cell = LSTM_Cell(args,
                                                  input_size=args.rnn_size + args.conv_out_size * args.conv_out_channels,
                                                  output_size=args.aux_rnn_size,
                                                  scope_name="attn_lstm_{0}".format(i),
                                                  current_weights=attention_W1[0])
                        else:
                            temp_cell = LSTM_Cell(args,
                                                  input_size=args.aux_rnn_size,
                                                  output_size=args.aux_rnn_size,
                                                  scope_name="attn_lstm_{0}".format(i),
                                                  current_weights=attention_W[i-1])
                        self.attention_cell_list.append(temp_cell)
                        self.attention_cells = tf.contrib.rnn.MultiRNNCell(self.attention_cell_list)
                        self.attention_initial_state = self.attention_cells.zero_state(args.batch_size, tf.float32)


                if args.weight_noise_type == "adaptive":
                    print("Initializing ReLU cells; require {0} task 1 weights                             and {1} task 2 weights.".format(task_1_weight_size, task_2_weight_size))
                else:
                    task_1_W1 = [None]
                    task_1_W = [None for i in range(args.task_1_layers)]
                    task_2_W1 = [None]
                    task_2_W = [None for i in range(args.task_2_layers)]
                
                self.task_1_cell_list = []
                for i in range(args.task_1_layers):
                    if i==0:
                        temp_cell = ReLU_Cell(args,
                                              input_size=args.rnn_size,
                                              scope_name="task_1_relu_{0}".format(i),
                                              current_weights=task_1_W1[0])
                    else:
                        if args.lateral_connections:
                            temp_cell = ReLU_Cell(args,
                                                  input_size=args.relu_size*2,
                                                  scope_name="task_1_relu_{0}".format(i),
                                                  current_weights=task_1_W[i-1])
                        else:
                            temp_cell = ReLU_Cell(args,
                                                  input_size=args.relu_size,
                                                  scope_name="task_1_relu_{0}".format(i),
                                                  current_weights=task_1_W[i-1])
                    self.task_1_cell_list.append(temp_cell)

                self.task_2_cell_list = []

                for i in range(args.task_2_layers):
                    if i==0:
                        temp_cell = ReLU_Cell(args,
                                              input_size=args.rnn_size,
                                              scope_name="task_2_relu_{0}".format(i),
                                              current_weights=task_2_W1[0])
                    else:
                        if args.lateral_connections:
                            temp_cell = ReLU_Cell(args,
                                                  input_size=args.relu_size*2,
                                                  scope_name="task_2_relu_{0}".format(i),
                                                  current_weights=task_2_W[i-1])
                        else:
                            temp_cell = ReLU_Cell(args,
                                                  input_size=args.relu_size,
                                                  scope_name="task_2_relu_{0}".format(i),
                                                  current_weights=task_2_W[i-1])
                    self.task_2_cell_list.append(temp_cell)

             
        with tf.device("/gpu:0"):
            if args.use_embeddings:
                temp_x = tf.transpose(self.x)
                temp_x = tf.reshape(temp_x, [args.char_vocab_len, args.max_example_string_len * args.batch_size])
                embedding_output = tf.matmul(self.embedding_mat, tf.cast(temp_x, tf.float32))
                embedding_output = tf.reshape(embedding_output,
                                              [1, args.embedding_size, args.max_example_string_len, args.batch_size])
                embedding_output = tf.transpose(embedding_output)

    #             raw_inputs = tf.split(axis=1, num_or_size_splits=self.max_example_string_len, value=embedding_output)
    #             raw_inputs = [tf.squeeze(x, [1]) for x in raw_inputs]
                if args.use_convolution: 
                    rnn_inputs = tf.nn.conv2d(embedding_output, self.conv_W,
                                              [1, args.conv_stride, 1, 1], padding=args.conv_padding, data_format="NHWC")
                else:
                    rnn_inputs = tf.squeeze(embedding_ouput)

            else:
                if args.use_convolution: 
                    rnn_inputs = tf.nn.conv1d(self.x, self.conv_W, args.conv_stride,
                                              padding=args.conv_padding, data_format="NHWC")

                else:
                    rnn_inputs = tf.cast(self.x, tf.float32)

            print("Initializing LSTM call.")
            shared_output, last_state = self.shared_cells(rnn_inputs, self.shared_initial_state)
            
            if args.use_aux_task:
                rnn_input_rs = tf.reshape(rnn_inputs, [args.batch_size, -1])
                attention_input = tf.concat([shared_output, rnn_input_rs], 1)
                attn_cell_output, attention_state = self.attention_cells(attention_input, self.attention_initial_state)
                attention_output = tf.matmul(attn_cell_output, self.attention_out_W) + self.aux_b
                self.aux_logits = tf.sigmoid(attention_output)

            shared_output = tf.reshape(tf.concat(shared_output,1), [-1, args.rnn_size])

            print("Initializing ReLU call.")

            if args.task_1_layers != args.task_2_layers:
                if args.lateral_connections:
                    raise Exception("Lateral connections not implemented for task 1 layers != task 2 layers.")
                for i in range(args.task_1_layers):
                    if i==0:
                        exec("self.task_1_cell_{0} = self.task_1_cell_list[{0}]".format(i))
                        exec("self.task_1_initial_state = self.task_1_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                        exec("task_1_outputs_{0}, task_1_state_{0} = self.task_1_cell_{0}(shared_output,                                                                                                 self.task_1_initial_state)".format(i))
                    else:
                        exec("self.task_1_cell_{0} = self.task_1_cell_list[{0}]".format(i))
                        exec("self.task_1_initial_state = self.task_1_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                        exec("task_1_outputs_{0}, task_1_state_{0} = self.task_1_cell_{0}(task_1_outputs_{1},                                                                                                 self.task_1_initial_state)".format(i, i-1))
                for i in range(args.task_2_layers):
                    if i==0:
                        exec("self.task_2_cell_{0} = self.task_2_cell_list[{0}]".format(i))
                        exec("self.task_2_initial_state = self.task_2_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                        exec("task_2_outputs_{0}, task_2_state_{0} = self.task_2_cell_{0}(shared_output,                                                                                                 self.task_2_initial_state)".format(i))
                    else:
                        exec("self.task_2_cell_{0} = self.task_2_cell_list[{0}]".format(i))
                        exec("self.task_2_initial_state = self.task_2_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                        exec("task_2_outputs_{0}, task_2_state_{0} = self.task_2_cell_{0}(task_2_outputs_{1},                                                                                                 self.task_2_initial_state)".format(i, i-1))
            else:
                if args.lateral_connections:
                    for i in range(args.task_1_layers):
                        if i==0:
                            exec("self.task_1_cell_{0} = self.task_1_cell_list[{0}]".format(i))
                            exec("self.task_1_initial_state = self.task_1_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                            exec("task_1_outputs_{0}, task_1_state_{0} = self.task_1_cell_{0}(shared_output,                                                                                                     self.task_1_initial_state)".format(i))
                            exec("self.task_2_cell_{0} = self.task_2_cell_list[{0}]".format(i))
                            exec("self.task_2_initial_state = self.task_2_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                            exec("task_2_outputs_{0}, task_2_state_{0} = self.task_2_cell_{0}(shared_output,                                                                                                 self.task_2_initial_state)".format(i))
                        else:
                            exec("self.task_1_cell_{0} = self.task_1_cell_list[{0}]".format(i))
                            exec("self.task_1_initial_state = self.task_1_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                            exec("task_1_inputs_{0} = tf.concat([task_1_outputs_{1}, task_2_outputs_{1}], 1)".format(i, i-1))
                            exec("task_1_outputs_{0}, task_1_state_{0} = self.task_1_cell_{0}(task_1_inputs_{0},                                                                                                     self.task_1_initial_state)".format(i))
                            exec("self.task_2_cell_{0} = self.task_2_cell_list[{0}]".format(i))
                            exec("self.task_2_initial_state = self.task_2_cell_{0}.zero_state(args.batch_size, tf.float32)".format(i))
                            exec("task_2_inputs_{0} = tf.concat([task_2_outputs_{1}, task_1_outputs_{1}], 1)".format(i, i-1))
                            exec("task_2_outputs_{0}, task_2_state_{0} = self.task_2_cell_{0}(task_2_inputs_{0},                                                                                                     self.task_2_initial_state)".format(i))

            exec("self.task_1_outputs = task_1_outputs_{}".format(len(self.task_1_cell_list)-1))
            exec("self.task_2_outputs = task_2_outputs_{}".format(len(self.task_2_cell_list)-1))

            self.task_1_logits = tf.matmul(self.task_1_outputs, self.c_sfmx_W) + self.task_1_b
            self.task_2_logits = tf.matmul(self.task_2_outputs, self.b_sfmx_W) + self.task_2_b
        
    def train(self, args):
        with tf.device("/gpu:0"):
            if args.weight_noise_type is None:
                if args.use_aux_task:
                    self.loss = -log_likelihood(y1=self.y1_, y2=self.y2_, aux_y=self.aux_y, aux_logit=self.aux_logits,
                                           logit1=self.task_1_logits, logit2=self.task_2_logits, args=args)
                else:
                    self.loss = -log_likelihood(y1=self.y1_, y2=self.y2_, logit1=self.task_1_logits,
                                           logit2=self.task_2_logits, args=args)
            else:
                if args.use_aux_task:
                    self.elbo = tf.contrib.bayesflow.variational_inference.elbo(
                        log_likelihood(y1=self.y1_, y2=self.y2_, aux_y=self.aux_y, aux_logit=self.aux_logits,
                                       logit1=self.task_1_logits, logit2=self.task_2_logits, args=args),
                        {self.weight_st: self.weight_prior},
                        keep_batch_dim=True
                    )
                    self.loss = -self.elbo
                else:
                    self.elbo = tf.contrib.bayesflow.variational_inference.elbo(
                        log_likelihood(y1=self.y1_, y2=self.y2_, logit1=self.task_1_logits,
                                       logit2=self.task_2_logits, args=args),
                        {self.weight_st: self.weight_prior},
                        keep_batch_dim=True
                    )
                    self.loss = -self.elbo

            tf.add_to_collection('losses', self.loss)
            tf.add_n(tf.get_collection('losses'), name='total_loss')

            opt = tf.train.AdamOptimizer(args.learning_rate)

            if args.weight_noise_type in [None, "static"]:
                grads = opt.compute_gradients(self.loss)
                trunc_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in grads]
                apply_gradient_op = [opt.apply_gradients(trunc_grads, global_step=global_step)]

                for grad, var in trunc_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/gradients', grad)

            else: 
                with tf.variable_scope("weights", reuse=True):
                    means = tf.get_variable("mu")
                    variances =  tf.get_variable("S_hat")
                    biases = [tf.get_variable("task_1_b"), tf.get_variable("task_2_b")]
                    for i in range(args.shared_layers):
                        with tf.variable_scope("shared_lstm_{0}".format(i), reuse=True):
                            biases.append(tf.get_variable("b"))
                    for i in range(args.task_1_layers):
                        with tf.variable_scope("task_1_relu_{0}".format(i), reuse=True):
                            biases.append(tf.get_variable("b"))
                    for i in range(args.task_2_layers):
                        with tf.variable_scope("task_2_relu_{0}".format(i), reuse=True):
                            biases.append(tf.get_variable("b")) 

                var_grads = opt.compute_gradients(self.loss, var_list=[variances])
                self.trunc_var_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in var_grads]
                apply_var_grad_op = opt.apply_gradients(self.trunc_var_grads, global_step=global_step)

                mean_grads = opt.compute_gradients(self.loss, var_list=[means])
                self.trunc_mean_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in mean_grads]
                apply_mean_grad_op = opt.apply_gradients(self.trunc_mean_grads, global_step=global_step)

                self.bias_grads = opt.compute_gradients(self.loss, var_list=biases)
                apply_bias_grad_op = opt.apply_gradients(self.bias_grads, global_step=global_step)

                apply_gradient_op = [apply_mean_grad_op] + [apply_var_grad_op] + [apply_bias_grad_op]

                for grad, var in self.trunc_mean_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/mean_gradients', grad)
                for grad, var in self.trunc_var_grads:
                    if grad is not None:
                        tf.summary.histogram(var.op.name + '/variance_gradients', grad)

            for var in tf.global_variables():
                tf.summary.histogram(var.op.name, var)

            if args.compute_variable_averages:
                moving_averager = tf.train.ExponentialMovingAverage(0.9, name='avg')
                variables_averages_op = moving_averager.apply(tf.trainable_variables())
                _add_loss_summaries(model.loss, moving_averager, args.compute_variable_averages)

                with tf.control_dependencies(apply_gradient_op + [variables_averages_op]):
                    self.train_op = tf.no_op(name='train')
            else:
                _add_loss_summaries(model.loss)
                with tf.control_dependencies(apply_gradient_op):
                    self.train_op = tf.no_op(name='train')

            return self.train_op

    def evaluate(self, args):
        if args.weight_noise_type is None:
            if not hasattr(self, 'eval_loss'):
                self.eval_loss = []
            if args.use_aux_task:
                self.eval_loss.append(
                    -log_likelihood(y1=self.y1_, y2=self.y2_, aux_y=self.aux_y, aux_logit=self.aux_logits,
                                       logit1=self.task_1_logits, logit2=self.task_2_logits, args=args)
                )
            else:
                self.eval_loss.append(
                    -log_likelihood(y1=self.y1_, y2=self.y2_, logit1=self.task_1_logits,
                                       logit2=self.task_2_logits, args=args)
                )
                
        else:
            if not hasattr(self, 'eval_loss'):
                self.eval_loss = []
            if args.use_aux_task:
                self.eval_loss.append(
                    -tf.contrib.bayesflow.variational_inference.elbo(
                        log_likelihood(y1=self.y1_, y2=self.y2_, aux_y=self.aux_y, aux_logit=self.aux_logits,
                                       logit1=self.task_1_logits, logit2=self.task_2_logits, args=args),
                        {self.weight_st: self.weight_prior},
                        keep_batch_dim=True
                    )
                )
            else:
                self.eval_loss.append(
                    -tf.contrib.bayesflow.variational_inference.elbo(
                        log_likelihood(y1=self.y1_, y2=self.y2_, logit1=self.task_1_logits,
                                       logit2=self.task_2_logits, args=args),
                        {self.weight_st: self.weight_prior},
                        keep_batch_dim=True
                    )
                )
            
        if not hasattr(self, 'label1_accuracy'):
            self.label1_accuracy = []
        if not hasattr(self, 'label2_accuracy'):
            self.label2_accuracy = []
        
        self.label1_accuracy.append(tf.equal(tf.argmax(self.y1_, 1), tf.argmax(self.task_1_logits, 1)))
        self.label2_accuracy.append(tf.equal(tf.argmax(self.y2_, 1), tf.argmax(self.task_2_logits, 1)))
        
        logit1_bool = tf.one_hot(tf.argmax(self.task_1_logits, 1), depth=args.task_1_vocab_len)
        self.y1_TP = tf.count_nonzero(logit1_bool * self.y1_)
        self.y1_TN = tf.count_nonzero((logit1_bool - 1) * (self.y1_ - 1))
        self.y1_FP = tf.count_nonzero(logit1_bool * (self.y1_ - 1))
        self.y1_FN = tf.count_nonzero((logit1_bool - 1) * self.y1_)
                           
        logit2_bool = tf.one_hot(tf.argmax(self.task_2_logits, 1), depth=args.task_2_vocab_len)
        self.y2_TP = tf.count_nonzero(logit2_bool * self.y2_)
        self.y2_TN = tf.count_nonzero((logit2_bool - 1) * (self.y2_ - 1))
        self.y2_FP = tf.count_nonzero(logit2_bool * (self.y2_ - 1))
        self.y2_FN = tf.count_nonzero((logit2_bool - 1) * self.y2_)
        
        class_metrics = {'y1_TP': tf.cast(self.y1_TP, tf.float32),
                         'y1_TN': tf.cast(self.y1_TN, tf.float32),
                         'y1_FP': tf.cast(self.y1_FP, tf.float32),
                         'y1_FN': tf.cast(self.y1_FN, tf.float32),
                         'y2_TP': tf.cast(self.y2_TP, tf.float32),
                         'y2_TN': tf.cast(self.y2_TN, tf.float32),
                         'y2_FP': tf.cast(self.y2_FP, tf.float32),
                         'y2_FN': tf.cast(self.y2_FN, tf.float32)}
        
        return [self.eval_loss, self.label1_accuracy, self.label2_accuracy, class_metrics]
    
    def sample(self):
        flat_y1_ = tf.expand_dims(tf.reshape(self.y1_, [-1]), 1)
        flat_y1_ = tf.cast(flat_y1_, tf.int64)
        softmax1 = tf.nn.softmax(self.task_1_logits)
        samples1 = tf.multinomial(softmax1, 1)
        
        flat_y2_ = tf.expand_dims(tf.reshape(self.y2_, [-1]), 1)
        flat_y2_ = tf.cast(flat_y2_, tf.int64)
        softmax2 = tf.nn.softmax(self.task_2_logits)
        samples2 = tf.multinomial(softmax2, 1)
        
        self.sampled_results = tf.concat([flat_y1_, samples1, flat_y2_, samples2], axis=0)
        return self.sampled_results


# In[16]:

data_path = "data/"

train_path = data_path + "csv_datasets/train.csv"
dev_path = data_path + "csv_datasets/dev.csv"

model_path = 'baseLSTM'


# In[17]:

example_vocab = pd.read_csv(data_path + 'vocab/train_example_vocab_0_2042600_no_numeric_only.csv',
                          header=None, encoding="ISO-8859-1")
example_vocab = example_vocab.reset_index()
example_vocab.index = example_vocab.ix[:, 1].str.encode('utf-8')

example_vocab = example_vocab.ix[:, 0]
example_vocab_len = len(example_vocab)

example_vocab = example_vocab.to_dict()


# In[18]:

task_2_vocab = pd.read_csv(data_path + 'vocab/task_2_vocab_0_179322.csv', header=None, encoding="ISO-8859-1")
task_2_vocab = task_2_vocab.reset_index()
task_2_vocab.index = task_2_vocab.ix[:, 1].str.encode('utf-8')

task_2_vocab = task_2_vocab.ix[:, 0]
task_2_vocab_len = len(task_2_vocab)

task_2_vocab = task_2_vocab.to_dict()


# In[19]:

task_1_vocab = pd.read_csv(data_path + 'vocab/task_1_vocab_0_1728.csv', header=None, encoding="ISO-8859-1")
task_1_vocab = task_1_vocab.ix[:,0].str.lower().str.split().str.join("")
task_1_vocab = task_1_vocab.reset_index()
task_1_vocab.index = task_1_vocab.ix[:, 1].str.encode('utf-8')

task_1_vocab = task_1_vocab.ix[:, 0]
task_1_vocab_len = len(task_1_vocab)

task_1_vocab = task_1_vocab.to_dict()


# In[20]:

char_vocab = pd.read_csv(data_path + 'vocab/alphanumeric_vocab.csv', index_col=0)
char_vocab.index = char_vocab.ix[:, 1].str.encode('utf-8')

char_vocab = char_vocab.drop('character', axis=1)
char_vocab = char_vocab.ix[:, 0]

char_vocab_len = len(char_vocab)
char_vocab = char_vocab.to_dict()


# In[21]:

full_model_dir = model_path

if not os.path.exists(full_model_dir):
    os.makedirs(full_model_dir)


# In[22]:

class ArgStruct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


# In[23]:

arg_dict = {
    'data_path': data_path,
    'model_path': model_path,
#     'example_vocab': example_vocab,
#     'example_vocab_len': example_vocab_len+1,
    'task_2_vocab': task_2_vocab,
    'task_2_vocab_len': task_2_vocab_len+2,
    'task_1_vocab': task_1_vocab,
    'task_1_vocab_len': task_1_vocab_len+1,
    'character_level_training': True,
    'use_aux_task': True,
    'char_vocab': char_vocab,
    'char_vocab_len': char_vocab_len+1,
    'max_example_string_len': 150,
    'use_embeddings': True,
    'embedding_size': 300,
    'use_convolution': True,
    'conv_width': 5,
    'conv_stride': 1,
    'conv_padding': 'SAME',
    'conv_out_channels': 1,
    'sentence_rep': 'bow',
    'shuffle_batches': False,
    'rnn_size': 150,
    'aux_rnn_size': 150,
    'relu_size': 50,
    'lateral_connections': True,
    'shared_layers': 4,
    'aux_layers': 1,
    'task_1_layers': 2,
    'task_2_layers': 2,
    'batch_size': 1,
    'seq_length': 16,
    'forget_bias': 1.,
    'num_epochs': 1,
    'learning_rate': 5e-4,
    'momentum': 0.9,
    'logdir': 'TF_Logs',
    'save_every': 1000,
    'print_every': 50,
    'compute_variable_averages': False,
    'weight_noise_type': None,
    'weight_prior_variance': 0.075,
    'warm_start': False,
    'skip_example_rows': 0,
    'evaluation_time': False,
    'ps_tasks': 3,
    'max_iterations': 20000
}


# In[24]:

args = ArgStruct(**arg_dict)


# In[25]:

# greedy = tf.contrib.training.GreedyLoadBalancingStrategy(num_tasks=args.ps_tasks,
#                                                          load_fn=tf.contrib.training.byte_size_load_fn)

sess_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True


# In[26]:

with tf.Graph().as_default():
    global global_step
    global_step = tf.Variable(0, name='global_step', trainable=False)

    example_feed, label1_feed, label2_feed, aux_label_feed = input_pipeline(
        [train_path],
        args=args)

    dev_example_feed, dev_label1_feed, dev_label2_feed, dev_aux_label_feed = input_pipeline(
        [dev_path],
        args=args)

    with tf.Session(config=sess_config).as_default() as sess:
#         with tf.device(tf.train.replica_device_setter(ps_tasks=args.ps_tasks)):

        print("Adding model to graph.")
        model = Model(args=args)

        print("Adding training and sampling ops to graph.")
        train_op = model.train(args=args)
        sample_op = model.sample()
        metric_op = model.evaluate(args=args)

        writer = tf.summary.FileWriter(args.logdir, sess.graph)

#         if args.use_embeddings:
#             proj_config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()

#             embedding = proj_config.embeddings.add()
#             embedding.tensor_name = model.embedding_mat.name
#             embedding.metadata_path = os.path.join(args.logdir, 'metadata.tsv')

#             tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, proj_config)

        merged = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)


        saver = tf.train.Saver()
        if not args.warm_start:
            saver.save(sess, os.path.join(args.model_path, "model.ckpt"), global_step.eval())

        else:
            print(tf.train.latest_checkpoint('./' + args.model_path))
            saver.restore(sess, tf.train.latest_checkpoint('./' + args.model_path))


        coord = tf.train.Coordinator()  
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        global queue_stopped
        queue_stopped = coord.should_stop()

        args.evaluation_time = False

        while not coord.should_stop():
            if not args.evaluation_time:
                try:
                    start_time = time.time()                
                    example_batch, label1_batch, label2_batch, aux_label_batch = sess.run([example_feed,
                                                                          label1_feed,
                                                                          label2_feed,
                                                                          aux_label_feed])

                    run_options = tf.RunOptions(output_partition_graphs=True,
                                                trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    result, summary = sess.run(
                        [train_op, merged],
                        feed_dict={model.x: example_batch,
                                   model.y1_: label1_batch,
                                   model.y2_: label2_batch,
                                   model.aux_y: aux_label_batch},
                        options=run_options,
                        run_metadata=run_metadata
                    )
                    
                    if args.weight_noise_type is None:
                        glob_step = global_step.eval()
                    elif arg.weight_noise_type in ('static', 'adaptive'):
                        glob_step = global_step.eval()//3
                    
                    if (glob_step) % args.print_every == 0:
                        if (glob_step)==1:
                            print(run_metadata)
                        writer.add_summary(summary, global_step.eval())
                        writer.add_run_metadata(run_metadata,
                                            tag="step{0}".format(global_step.eval()),
                                            global_step=global_step.eval())
                        
                        if args.weight_noise_type is None:
                            latest_loss = sess.run([model.loss],
                                                   feed_dict={model.x: example_batch,
                                                              model.y1_: label1_batch,
                                                              model.y2_: label2_batch,
                                                              model.aux_y: aux_label_batch})
                            try:
                                summary_nums = (glob_step, np.mean(duration), np.mean(latest_loss))
                                print('Iteration: {0}; Last Step Duration: {1}; Loss: {2}.'.format(*summary_nums))
                            except:
                                pass
                        elif args.weight_noise_type in ('static', 'adaptive'):
                            latest_elbo, latest_loss = sess.run([model.elbo, model.loss],
                                                   feed_dict={model.x: example_batch,
                                                              model.y1_: label1_batch,
                                                              model.y2_: label2_batch,
                                                              model.aux_y: aux_label_batch})
                            try:
                                summary_nums = (glob_step, duration,
                                                np.mean(latest_elbo), np.mean(latest_loss))
                                print('Iteration: {0}; Last Step Duration: {1}; ELBO: {2}; Loss: {3}.'.format(*summary_nums))
                            except:
                                pass

                    # Save the model and the vocab
                    if glob_step % args.save_every == 0:
                        if glob_step > 0:
                            clear_output()
                        # Save model
                        model_file_name = os.path.join(full_model_dir, 'model.ckpt')
                        saver.save(sess, model_file_name, global_step=global_step)
                        print('Model Saved To: {}'.format(model_file_name))


                    duration = time.time() - start_time

                    writer.flush()
                    
                    if global_step.eval()//3 >= args.max_iterations:
                        raise tf.errors.OutOfRangeError(tf.no_op("Stopper"), "Stopper", message="Max iterations reached.")
                    
                except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                    args.evaluation_time = True
                    evaluation_step = 0
                    print('Done training for %d epochs, %d steps.' % (args.num_epochs, global_step.eval()))
                    print("Evaluating against dev set...")
                    
                    y1_TP = 0.
                    y1_TN = 0.
                    y1_FP = 0.
                    y1_FN = 0.
                    y2_TP = 0.
                    y2_TN = 0.
                    y2_FP = 0.
                    y2_FN = 0.

            else:
                try:
                    dev_example_batch, dev_label1_batch, dev_label2_batch, dev_aux_label_batch = sess.run(
                        [dev_example_feed,
                         dev_label1_feed,
                         dev_label2_feed,
                         dev_aux_label_feed])
                    eval_metrics = sess.run(metric_op,
                                            feed_dict={model.x: dev_example_batch,
                                                       model.y1_: dev_label1_batch,
                                                       model.y2_: dev_label2_batch,
                                                       model.aux_y: aux_label_batch})
                    y1_TP += eval_metrics[3].get('y1_TP')
                    y1_TN += eval_metrics[3].get('y1_TN')
                    y1_FP += eval_metrics[3].get('y1_FP')
                    y1_FN += eval_metrics[3].get('y1_FN')
                    y2_TP += eval_metrics[3].get('y2_TP')
                    y2_TN += eval_metrics[3].get('y2_TN')
                    y2_FP += eval_metrics[3].get('y2_FP')
                    y2_FN += eval_metrics[3].get('y2_FN')

                    if evaluation_step % args.print_every == 0:
                        print('Step {0}'.format(evaluation_step))
                    
                    evaluation_step +=1

                except (tf.errors.OutOfRangeError, tf.errors.InvalidArgumentError) as e:
                    print("Evaluation complete. Calculating evaluation metrics.")                    
                    del eval_metrics[3]

                    eval_metrics = [np.mean(x) for x in eval_metrics]

                    print("Average loss on validation set: {0} \n"
                          "Label1 Accuracy: {1} \n"
                          "Label2 Accuracy: {2} \n".format(eval_metrics[0],
                                                           eval_metrics[1],
                                                           eval_metrics[2]))

                    y1_precision = y1_TP / (y1_TP + y1_FP)
                    y1_recall = y1_TP / (y1_TP + y1_FN)
                    y1_f1_score = (2 * y1_precision * y1_recall) / (y1_precision + y1_recall)

                    y2_precision = y2_TP / (y2_TP + y2_FP)
                    y2_recall = y2_TP / (y2_TP + y2_FN)
                    y2_f1_score = (2 * y2_precision * y2_recall) / (y2_precision + y2_recall)

                    print("task 1 Precision: {0} \n"
                          "task 1 Recall: {1} \n"
                          "task 1 F1 Score: {2} \n"
                          "task 2 Precision: {3} \n"
                          "task 2 Recall: {4} \n"
                          "task 2 F1 Score: {5} \n".format(y1_precision,
                                                          y1_recall,
                                                          y1_f1_score,
                                                          y2_precision,
                                                          y2_recall,
                                                          y2_f1_score))

                    # When done, ask the threads to stop.
                    coord.request_stop()
                queue_stopped = coord.should_stop()

        coord.join(threads)
        sess.close()


# In[ ]:




# In[ ]:



