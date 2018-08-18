import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
from _Functions_gesture import load_gestures, batch_generator
from layers import Layers

class Modell(Layers):
    
    
    def __init__(self, random_seed = 123,):
        self.root_logdir = '../logs/'
        self.modeldir = '../model/'
        self.now = datetime.utcnow().strftime('%Y%m%d%H%M%s')
        self.logdir = '{}/run-{}/'.format(self.root_logdir, self.now) 
        tf.set_random_seed(random_seed)
            
    def save(self, saver, sess, epoch, path='./model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        print('Modell speichern in {}'.format(path))
        saver.save(sess, os.path.join(path,'cnn-model.ckpt'), global_step=epoch)        
        
    def train(self, sess, training_set, validation_set=None, initialize=True, epochs=20, shuffle=True, dropout=0.5, random_seed=None):
        
        file_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
        start = time.time()        
        X_data = np.array(training_set[0])    
        y_data = np.array(training_set[1])
        
        if initialize:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
        else: 
            last = sorted(os.listdir(self.modeldir))[-1]
            meta_last = [i for i in os.listdir(self.modeldir+last) if 'meta' in i][0]
            epoch_last = int(meta_last.split('t-')[-1].split('.')[0])
            saver = tf.train.Saver()#tf.train.import_meta_graph(self.modeldir+last+'/{}'.format(meta_last))
            print('Modell laden aus {}'.format(os.path.join(self.modeldir+last)))
            saver.restore(sess, os.path.join(self.modeldir+last, 'cnn-model.ckpt-%d' % epoch_last))  

        for epoch in range(1, epochs+1):
            batch_gen = batch_generator(X_data, y_data, shuffle=shuffle, batch_size=64) 
            avg_loss = 0.0
            
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x, 'tf_y:0': batch_y, 'fc_keep_prob:0': dropout}
                loss,_ = sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed) 
                avg_loss += loss
             
            summary = tf.Summary()
            summary.value.add(tag="AVG_LOSS", simple_value=avg_loss)
            file_writer.add_summary(summary, epoch)    
            print('Epoch %02d Training LOSS: %7.3f' % (epoch, avg_loss), end=' ')
            #TODO
            # validation part doesn't work yet
            if validation_set is not None:
                feed = {'tf_x:0': validation_set[0],
                        'tf_y:0': validation_set[1],
                        'fc_keep_prob:0': 1.0}
                valid_acc = sess.run('accuracy:0', feed_dict=feed)
                print('KKR Validierung: %7.3f ' % valid_acc, end=' ')
                end_epoch_time = (time.time()-start)/60
                print('Time: {:.2f}min'.format(end_epoch_time))
                
                s_a, s_l = sess.run(['ACCURACY:0','LOSS:0'],feed_dict=feed)
                print(s_l)
                file_writer.add_summary(s_a, epoch)
                file_writer.add_summary(s_l, epoch)
                
            else: print()
        end = time.time() - start
        print('It took {:.2f}min'.format(end/60))
        file_writer.close()
        os.mkdir(self.modeldir+self.now)
        self.save(saver, sess, epochs, path=self.modeldir+self.now)


    def predict(self, sess, X_test, return_proba=False):
        feed = {'tf_x:0': X_test,
                'fc_keep_prob:0': 1.0}
        if return_proba:
            return sess.run('probabilities:0', feed_dict=feed)
        else:
            return sess.run('labels:0', feed_dict=feed)
        
if __name__=='__main__':

#%% Data for training
    X_train, y_train = load_gestures(path='../../gestures/train')
    X_valid, y_valid = load_gestures(path='../../gestures/test')
    X_valid, y_valid = X_valid[:64], y_valid[:64]
    
    
    
#%% CNN model
    
    model = Modell()
    learning_rate = 1e-4
    
 
    g = tf.Graph()
    with g.as_default():
        
        model.build_cnn(learning_rate)
        



        
    with tf.Session(graph=g) as sess:
        
        summary = model.train(sess, 
                    training_set=(X_train,y_train),
                    validation_set=None, #####            (X_valid, y_valid)
                    initialize=False,
                    random_seed=123, 
                    epochs=5)
#        model.save(saver,sess,epoch=5, path='../model_test/')
#        preds = predict(sess, X_valid, return_proba=False)
#        print('KKR Test: {:.3f}%'.format((100*np.sum(preds==y_valid)/len(y_valid))))    
        













































    
    