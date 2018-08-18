import tensorflow as tf
import numpy as np

class Layers():
    """Diese Classe stellt die Convolutional und die Full-Conectet Schichten zur Verfügung 
    und mit der Methode 'build_cnn' wird der Graph gebaut."""    
        
    def _conv_layer(self, input_tensor, name, kernel_size, n_output_channels, padding_mode='SAME', strides=(1,1,1,1)):
        
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
    
    def _fc_layer(self, input_tensor, name, n_output_units, activation_fn=None):
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
        
    def build_cnn(self,learning_rate):
        tf_x = tf.placeholder(tf.float32, shape=[None, 71500], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=[None], name='tf_y')    
        tf_x_image = tf.reshape(tf_x, shape=[-1,275,260,1], name='tf_reshaped')
        tf_y_onehot = tf.one_hot(indices=tf_y, depth=6, dtype=tf.float32, name='tf_y_onehot')

        print('\nErste Schicht: Faltung_1') # 1. Faltungsschicht & Max-Pooling
        h1 = self._conv_layer(tf_x_image, name= 'conv_1', kernel_size=(5,5), n_output_channels=32, padding_mode='VALID')
        h1_pool = tf.nn.max_pool(h1, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
        
        print('\nZweite Schicht: Faltung_2') # 2. Faltungsschicht & Max-Pooling
        h2 = self._conv_layer(h1_pool, name= 'conv_2', kernel_size=(5,5), n_output_channels=48, padding_mode='VALID')
        h2_pool = tf.nn.max_pool(h2, ksize=[1,3,3,1], strides=[1,3,3,1], padding='SAME')
    
        print('\nZwischen Schicht: Faltung_3') # 3. Faltungsschicht & Max-Pooling
        h3 = self._conv_layer(h2_pool, name= 'conv_3', kernel_size=(5,5), n_output_channels=64, padding_mode='VALID')
        h3_pool = tf.nn.max_pool(h3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        print('\nDritte Schicht: Vollständig verknüpft')
        h3 = self._fc_layer(h3_pool, name='fc_3', n_output_units=1000, activation_fn=tf.nn.relu)
        keep_prob = tf.placeholder(tf.float32, name='fc_keep_prob')
        h3_drop = tf.nn.dropout(h3, keep_prob=keep_prob, name='dropout_layer')
        
        print('\nVierte Schicht: Vollständig verknüpft -lineare aktivierung-')
        h4 = self._fc_layer(h3_drop, name='fc_4', n_output_units=6, activation_fn=None)
        
        # Vorhersage
        predictions = {'probabilities': tf.nn.softmax(h4, name='probabilities'),
                       'labels': tf.cast(tf.argmax(h4, axis=1), dtype=tf.int32, name='labels')}
        
        ## Visualisierung des Graphen mit TensorBoard & Verlustfunktion und Optimierung
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y_onehot), name='cross_entropy_loss')
        
        # Optimierung:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
        
        # Berechnung der Korrektklassifizierungsrate
        correct_predictions = tf.equal(predictions['labels'], tf_y, name='correct_preds')
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')

