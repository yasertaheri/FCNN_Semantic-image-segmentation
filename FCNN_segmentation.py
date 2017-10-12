# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 00:58:28 2017

@author: Yaser M.Taheri
"""
"""
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow
import pandas as pd
from PIL import Image
from FCNN_Network import modified_VGG

NUM_CLASS = 21
BATCH_SZIE = 1     

####################################################################################################################################
####################################################################################################################################             
             
tf.reset_default_graph()  

X = tf.placeholder( tf.float32, shape = [1,None,None,3] )         # input RGB image
Y = tf.placeholder( tf.int64, shape = [1,None,None,1] )           # Groubd truth
      
net=modified_VGG(X)

####################################################################################################################################
   
list=pd.read_csv("C:/VOCdevkit/VOC2012/ImageSets/Segmentation/train.csv", index_col= None, header = None, names = ['train_set'])

annotation_pred = tf.argmax(net.conv14, dimension=3, name="prediction")
annotation_pred=tf.expand_dims(annotation_pred,3)

tf.summary.image("input_image", X)
tf.summary.image("ground_truth", tf.cast(Y,tf.float32))
tf.summary.image("pred_annotation", tf.cast(annotation_pred, tf.float32))

correct_p= tf.equal(annotation_pred,Y )

Acuraccy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
tf.summary.scalar("Acuttacy", Acuraccy)

#target=tf.one_hot(tf.squeeze(Y,3),21)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=net.conv14, labels = tf.squeeze(Y,3)))


#epsilon = tf.constant(1e-12)
#out = tf.nn.softmax(vgg.conv14) + epsilon
#loss = (tf.reduce_sum(-target*tf.log(out)))/ (375*500)

tf.summary.scalar("loss", loss)

optimizer=tf.train.AdamOptimizer(0.0001).minimize(loss)
    
init = tf.global_variables_initializer() 

summary_op = tf.summary.merge_all()

saver=tf.train.Saver()
       
with tf.Session() as sess:
    
     sess.run(init)
         
##################################################################################################################################
#####################################################  weight initialization #####################################################
     c=np.load('vgg16_weights.npz')
     p=sorted(c.keys())    
     for i, k in enumerate(p):
         if i<26:
            sess.run(net.parameters[i].assign(c[k]))    
                 
         if i==26:
            sess.run(net.parameters[i].assign(tf.reshape(c[k],[7,7,512,4096])))    
         if i==27:
            sess.run(net.parameters[i].assign(tf.reshape(c[k],[4096])))    
         if i==28:
            sess.run(net.parameters[i].assign(tf.reshape(c[k],[1,1,4096,4096])))    
         if i==29:
            sess.run(net.parameters[i].assign(tf.reshape(c[k],[4096])))
        
################################################################################################################################
################################################################################################################################      
     
     tf.get_default_graph().finalize()
     writer = tf.summary.FileWriter('c:/newgraph', sess.graph)


     for epoch in range(100):
         
         for i in range(1,len(list)):
             
             image=imread("C:/VOCdevkit/VOC2012/JPEGImages/"+ list.loc[i,'train_set'] +".jpg")             #input image
             anot=Image.open("C:/VOCdevkit/VOC2012/SegmentationClass/"+ list.loc[i,'train_set'] +".png")   #annotation
             
             anot=np.array(anot,dtype=np.uint8)
             anot[anot==255] = 0
             annotation=np.expand_dims(anot, axis=0)
             annotation=np.expand_dims(annotation, axis=3)
             image=np.expand_dims(image, axis=0)
 
           
             _,l, a, an,sumery=sess.run([optimizer,loss, Acuraccy, annotation_pred,summary_op], feed_dict={X:image, Y:annotation})
          
             writer.add_summary(sumery, epoch)
             print("Epoch = " , epoch, "image_n = ", i, "loss=", l, "Accuracy = ", a)
             
         if epoch%5==0:
            save_path = saver.save(sess, "c:/newgraph/", epoch )
            print("Model saved in file: %s" % save_path)
         
