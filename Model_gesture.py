import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from Functions_gesture import load_gestures, batch_generator

#%% functions
def conv_layer(input_tensor, name, kernel_size, n_output_channels, padding_mode='SAME', strides=(1,1,1,1)):
    
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()
        n_input_channels = input_shape[-1]
        weights_shape = list(kernel_size) + [n_input_channels, n_output_channels]
        weights = tf.get_variable(name='_weights',shape=weights_shape)
        
        print(1, weights)        
        biases = tf.get_variable(name='_biases',initializer=tf.zeros(shape=[n_output_channels]))        
        print(2, biases)        
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=strides, padding=padding_mode)
        print(3, conv)
        conv = tf.nn.bias_add(conv, biases, name='net_pre-activation')
        print(4, conv)
        conv = tf.nn.relu(conv, name='activation')
        print(5, conv)
        
        return conv
    
    
def fc_layer(input_tensor, name, n_output_units, activation_fn=None):
    with tf.variable_scope(name):
        input_shape = input_tensor.get_shape().as_list()[1:]
        n_input_units = np.prod(input_shape)
        print('Input_Units',n_input_units)
        if len(input_shape) > 1:
            input_tensor = tf.reshape(input_tensor, shape=(-1, n_input_units))
        
        weights_shape = [n_input_units, n_output_units]
        weights = tf.get_variable(name='_weights', shape=weights_shape)
        print('weights:', weights)
        biases = tf.get_variable(name='_biases', initializer=tf.zeros(shape=[n_output_units]))
        print('biases:', biases)
        layer = tf.matmul(input_tensor, weights)
        print('layer:', layer)
        if activation_fn is None:
            return layer
        layer = activation_fn(layer, name='activation')
        print('layer_2:', layer)
        return layer
   
def build_cnn(learning_rate):
    tf_x = tf.placeholder(tf.float32, shape=[None, 71500], name='tf_x')
    # Images come in as flatted array in a batch: Shape=[batch, 784]
    tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')    
    tf_x_image = tf.reshape(tf_x, shape=[-1,275,260,1], name='tf_reshaped')
    tf_y_onehot = tf.one_hot(indices=tf_y, depth=6, dtype=tf.float32, name='tf_y_onehot')
        
    
    print('\nErste Schicht: Faltung_1')
    # 1. Faltungsschicht
    h1 = conv_layer(tf_x_image, name= 'conv_1', kernel_size=(5,5), n_output_channels=32, padding_mode='VALID')
    # 1. Max-Pooling
    h1_pool = tf.nn.max_pool(h1, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
    
#    print('\nZweite Schicht: Faltung_2')
#    # 2. Faltungsschicht
#    h2 = conv_layer(h1_pool, name= 'conv_2', kernel_size=(5,5), n_output_channels=48, padding_mode='VALID')
#    # 2. Max-Pooling
#    h2_pool = tf.nn.max_pool(h2, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')

    print('\nZwischen Schicht: Faltung_3')
    # 2. Faltungsschicht
    h3 = conv_layer(h1_pool, name= 'conv_3', kernel_size=(5,5), n_output_channels=64, padding_mode='VALID')
    # 2. Max-Pooling
    h3_pool = tf.nn.max_pool(h3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    
    print('\nDritte Schicht: Vollständig verknüpft')
    # 3. Schicht
    h3 = fc_layer(h3_pool, name= 'fc_3', n_output_units=1000, activation_fn=tf.nn.relu)
    keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
    h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')
    
    print('\nVierte Schicht: Vollständig verknüpft -lineare aktivierung-')
    h4 = fc_layer(h3_drop, name= 'fc_4', n_output_units=6, activation_fn=None)
    
    # Vorhersage
    predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                   'labels': tf.cast(tf.argmax(h4, axis=1), dtype=tf.int32, name='labels')}
    
    ## Visualisierung des Graphen mit TensorBoard
    ## Verlustfunktion und Optimierung
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot),
                                        name='cross_entropy_loss')
    
    # Optimierung:
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
    
    # Berechnung der Korrektklassifizierungsrate
    correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
    
def save(saver, sess, epoch, path='./model/'):
    if not os.path.isdir(path):
        os.makedirs(path)
    print('Modell speichern in {}'.format(path))
    saver.save(sess, os.path.join(path,'cnn-model.ckpt'), global_step=epoch)

def load(saver, sess, path, epoch):
    print('Modell laden aus {}'.format(path))
    saver.restore(sess, os.path.join(path, 'cnn-model.ckpt-%d' % epoch))

def train(sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True, dropout=0.5, random_seed=None):
    X_data = np.array(training_set[0])    
    y_data = np.array(training_set[1])
    training_loss = []
    
    if initialize:
        sess.run(tf.global_variables_initializer())
    
    np.random.seed(random_seed)
    start = time.time()
    for epoch in range(1, epochs+1):
        batch_gen = batch_generator(X_data, y_data, shuffle=shuffle) 
        avg_loss = 0.0
        
        for i, (batch_x, batch_y) in enumerate(batch_gen):
            feed = {'tf_x:0': batch_x,
                    'tf_y:0': batch_y,
                    'fc_keep_prob:0': dropout}
            loss,_ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
            avg_loss += loss
            
        training_loss.append(avg_loss/(i+1))
        print('Epoch %02d DWM Training: %7.3f' % (epoch, avg_loss), end=' ')
        
        if validation_set is not None:
            feed = {'tf_x:0': validation_set[0],
                    'tf_y:0': validation_set[1],
                    'fc_keep_prob:0': 1.0}
            valid_acc = sess.run('accuracy:0', feed_dict=feed)
            print('KKR Validierung: %7.3f ' % valid_acc, end=' ')
            print('Time: {:.2f}min'.format((time.time()-start)/60))
        else: print()
    end = time.time() - start
    print('It took {:.2f}min'.format(end/60))
    
def predict(sess, X_test, return_proba=False):
    feed = {'tf_x:0': X_test,
            'fc_keep_prob:0': 1.0}
    if return_proba:
        return sess.run('probabilities:0', feed_dict=feed)
    else:
        return sess.run('labels:0', feed_dict=feed)
    
if __name__=='__main__':
    
#%% Data for training
    X_data, y_data = load_gestures()

    X_test, y_test = load_gestures(path='./gestures_test/')
    
    X_train, y_train = X_data, y_data
    X_valid, y_valid = X_data[4369:,:], y_data[4369:]
#    
#    mean_vals = np.mean(X_train, axis=0)
#    std_val = np.std(X_train)
    
#    X_train_centered = (X_train-mean_vals)/std_val
#    X_test_centered = (X_test-mean_vals)/std_val
    
#%%  cnn-Model
    
    learning_rate = 1e-4
    random_seed = 123
    
#    g = tf.Graph()
#    with g.as_default():
#        tf.set_random_seed(random_seed)
#        build_cnn(learning_rate)
#        saver = tf.train.Saver()
#         
#    with tf.Session(graph=g) as sess:
#        train(sess, 
#              training_set=(X_train,y_train),
#              validation_set=(X_valid, y_valid),
#              initialize=True,
#              random_seed=123, 
#              epochs=7)
#        save(saver,sess,epoch=7)
#        preds = predict(sess, X_test, return_proba=False)
#        print('KKR Test: {:.3f}%'.format((100*np.sum(preds==y_test)/len(y_test))))    
#    
#    del g
#    
    g2 = tf.Graph()
    with g2.as_default():
        build_cnn(learning_rate)
        saver = tf.train.Saver()
    
    with tf.Session(graph=g2) as sess:
        load(saver, sess, epoch=20, path='./model/')
#        train(sess, 
#              training_set=(X_train,y_train),
#              validation_set=(X_valid, y_valid),
#              initialize=False,
#              random_seed=123)
#        save(saver,sess,epoch=20)
        preds = predict(sess, X_test, return_proba=False)
        print('KKR Test: {:.3f}%'.format((100*np.sum(preds==y_test)/len(y_test))))
        

    del g2
























