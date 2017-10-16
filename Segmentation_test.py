# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:57:02 2017

@author:  Yaser M. Taheri
"""

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize,imshow
import pandas as pd
from PIL import Image
from VGF_Segmentation import VGF
    
####################################################################################################################################
####################################################################################################################################             
             
tf.reset_default_graph()  

X = tf.placeholder( tf.float32, shape = [1,None,None,3] )  
Y = tf.placeholder( tf.int32, shape = [1,None,None,1] )  
      
vgg=VGF(X)


list=pd.read_csv("C:/Users/y_moh/Desktop/VOCtrainval_11-May-2012\VOCdevkit/VOC2012/ImageSets/Segmentation/train.csv", index_col= None, header = None, names = ['train_set'])

annotation_pred = (tf.argmax(vgg.conv14, dimension=3, name="prediction"))
cc=tf.argmax(Y,3)
correct_p= tf.equal(annotation_pred,cc )

Acuraccy = tf.reduce_mean(tf.cast(correct_p, tf.float32))

init = tf.global_variables_initializer() 
saver=tf.train.Saver()
a_t=0
with tf.Session() as sess:
     sess.run(init)
     saver.restore(sess,'c:\Segmentation\-2')

     for i in range(1,len(list)):
             
         image=imread("C:/Users/y_moh/Desktop/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/JPEGImages/"+ list.loc[i,'train_set'] +".jpg")
         a=Image.open("C:/Users/y_moh/Desktop/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/SegmentationClass/"+ list.loc[i,'train_set'] +".png")
         a=np.array(a,dtype=np.uint8)
         a[a==255] = 0
         annotation=np.expand_dims(a, axis=0)
         annotation=np.expand_dims(annotation, axis=3)
         image=np.expand_dims(image, axis=0)           
         ac=sess.run(Acuraccy, feed_dict= {X:image,Y:annotation})
         print ("imag N", i,"Accuracy_train = " , ac)
         a_t = ac + a_t
     print(a_t/(len(list)-1))    
