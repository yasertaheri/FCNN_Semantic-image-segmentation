import tensorflow as tf
import numpy as np


class modified_VGG:
    def __init__(self,imgs):
        self.imgs = imgs
        self.convlayers()
        #self.probs = tf.nn.softmax(self.fc3l)        
        #self.load_weights(weights, sess)
        
    def convlayers(self):
        self.parameters = []

 # zero-mean input
        with tf.variable_scope('preprocess'):
             mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1, 3], name='img_mean')
             image = self.imgs - mean  
             
        with tf.variable_scope('conv1_1'):
             W = tf.get_variable('weight', shape = [3,3,3,64], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [64], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(image,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv1_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv1_1/W",W)
             
        with tf.variable_scope('conv1_2'):
             W = tf.get_variable('weight', shape = [3,3,64,64], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [64], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv1_1 ,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv1_2 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv1_2/W",W)
             
                         
        with tf.name_scope('pool1'):
             self.pool1= tf.nn.max_pool(self.conv1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool1")
                          

        with tf.variable_scope('conv2_1'):
             W = tf.get_variable('weight', shape = [3,3,64,128], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool1,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv2_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv2_1/W",W)
             
        with tf.variable_scope('conv2_2'):
             W = tf.get_variable('weight', shape = [3,3,128,128], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [128], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv2_1,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv2_2 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv2_2/W",W)
             
        with tf.name_scope('pool2'):
            self.pool2= tf.nn.max_pool(self.conv2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool2")
                          
         
        with tf.variable_scope('conv3_1'):
             W = tf.get_variable('weight', shape = [3,3,128,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool2,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv3_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv3_1/W",W)
             
        with tf.variable_scope('conv3_2'):
              W = tf.get_variable('weight', shape = [3,3,256,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv3_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv3_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv3_2/W",W)
              
        with tf.variable_scope('conv3_3'):
              W = tf.get_variable('weight', shape = [3,3,256,256], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [256], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv3_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv3_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv3_3/W",W)
             
        with tf.name_scope('pool3'):
             self.pool3= tf.nn.max_pool(self.conv3_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool3")
             
                                       
        with tf.variable_scope('conv4_1'):
             W = tf.get_variable('weight', shape = [3,3,256,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool3,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv4_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv4_1/W",W)
             
             
        with tf.variable_scope('conv4_2'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv4_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv4_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv4_2/W",W)
              
              
        with tf.variable_scope('conv4_3'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv4_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv4_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv4_3/W",W)
             
        with tf.name_scope('pool4'):
             self.pool4= tf.nn.max_pool(self.conv4_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool4")
         
             
             
             
        with tf.variable_scope('conv5_1'):
             W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
             b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool4,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv5_1 = tf.nn.relu(conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("conv5_1/W",W)             

        with tf.variable_scope('conv5_2'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1),trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv5_1,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv5_2 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv5_2/W",W)
              
        with tf.variable_scope('conv5_3'):
              W = tf.get_variable('weight', shape = [3,3,512,512], initializer = tf.truncated_normal_initializer(stddev= 0.1), trainable=True)
              b = tf.get_variable('bias', shape = [512], initializer = tf.constant_initializer(0.0),trainable=True)
              conv = tf.nn.conv2d(self.conv5_2,W, strides = [1,1,1,1], padding = 'SAME')
              self.conv5_3 = tf.nn.relu(conv + b)
              self.parameters+=[W,b]
              tf.summary.histogram("conv5_3/W",W)
                                       
        with tf.name_scope('pool5'):
             self.pool5= tf.nn.max_pool(self.conv5_3, ksize = [1,2,2,1], strides = [1,2,2,1], padding= 'SAME', name = "pool5")
                   
             
        
        with tf.variable_scope('Fconv1'):
             
             W = tf.get_variable('weight', shape = [7, 7, 512, 4096], initializer = tf.truncated_normal_initializer(stddev= 0.01), trainable=True)
             b = tf.get_variable('bias', shape = [4096], initializer = tf.constant_initializer(0.0), trainable=True)
             conv = tf.nn.conv2d(self.pool5,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv11 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv1/W",W)
        
        with tf.variable_scope('Fconv2'):
             W = tf.get_variable('weight', shape = [1, 1, 4096, 4096], initializer = tf.truncated_normal_initializer(stddev= 0.01),trainable=True)
             b = tf.get_variable('bias', shape = [4096], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv11,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv12 = tf.nn.relu (conv + b)
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv2/W",W)
                      

        with tf.variable_scope('conv_score'):
             W = tf.get_variable('weight', shape = [1, 1, 4096, 21], initializer = tf.truncated_normal_initializer(stddev=(2 / 4096)**0.5),trainable=True)
             b = tf.get_variable('bias', shape = [21], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.conv12,W, strides = [1,1,1,1], padding = 'SAME')
             self.conv13 = conv + b
             self.parameters+=[W,b]
             tf.summary.histogram("Fconv3/W",W)
             
             
        with tf.variable_scope('up_pool4'):
            
             self.dshape1 = self.pool4.get_shape()
            
             weights= self.biliner_W( [4, 4, 21, 21])

             W = tf.get_variable('weight', shape = [4, 4, 21, 21], initializer =tf.constant_initializer(weights))             
             
             deconv_shape = tf.stack([tf.shape(self.pool4)[0], tf.shape(self.pool4)[1], tf.shape(self.pool4)[2], 21])
                                            
             conv = tf.nn.conv2d_transpose(self.conv13, W,deconv_shape, strides = [1,2,2,1], padding = 'SAME') 
             self.uppool4 = conv
             self.parameters+=[W]

        with tf.variable_scope('pool4_score'):
             W = tf.get_variable('weight', shape = [1, 1, 512, 21], initializer = tf.truncated_normal_initializer(stddev= 0.001),trainable=True)
             b = tf.get_variable('bias', shape = [21], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool4,W, strides = [1,1,1,1], padding = 'SAME')
             self.fus_pool4 = conv+ self.uppool4
             
                      
        with tf.variable_scope('up_pool3'):      
            
             self.dshape2 = self.pool3.get_shape()
                                            
             weights= self.biliner_W( [4, 4, 21, 21])
                                                 
             #W = tf.get_variable('weight', shape = [64, 64, 21, 21], initializer = tf.truncated_normal_initializer(stddev= 0.005),trainable=True)
             W = tf.get_variable('weight', shape = [4, 4, 21, 21], initializer = tf.constant_initializer(weights))
             #b = tf.get_variable('bias', shape = [self.dshape2[3].value], initializer = tf.constant_initializer(0.0),trainable=True) 
             
             #image_w = tf.shape(image)[1]
             #image_h = tf.shape(image)[2]
             deconv_shape = tf.stack([tf.shape(self.pool3)[0], tf.shape(self.pool3)[1], tf.shape(self.pool3)[2], 21])
                                            
             conv = tf.nn.conv2d_transpose(self.fus_pool4 ,W,deconv_shape, strides = [1,2,2,1], padding = 'SAME') 
             self.uppool3 = conv 
             self.parameters+=[W,b]
             tf.summary.histogram("FconvF/W",W)
             
        with tf.variable_scope('pool3_score'):
             W = tf.get_variable('weight', shape = [1, 1, 256, 21], initializer = tf.truncated_normal_initializer(stddev= 0.0001),trainable=True)
             b = tf.get_variable('bias', shape = [21], initializer = tf.constant_initializer(0.0),trainable=True)
             conv = tf.nn.conv2d(self.pool3,W, strides = [1,1,1,1], padding = 'SAME')
             self.fus_pool3 = conv+ self.uppool3
                          
             
             
        with tf.variable_scope('Final'):
            
             weights= self.biliner_W( [16, 16, 21, 21])
                       
             #W = tf.get_variable('weight', shape = [64, 64, 21, 21], initializer = tf.truncated_normal_initializer(stddev= 0.005),trainable=True)
             W = tf.get_variable('weight', shape = [16, 16, 21, 21], initializer = tf.constant_initializer(weights))
             #b = tf.get_variable('bias', shape = [21], initializer = tf.constant_initializer(0.0),trainable=True) 
             
             image_w = tf.shape(image)[1]
             image_h = tf.shape(image)[2]
             deconv_shape = tf.stack([1, image_w, image_h, 21])
                                            
             conv = tf.nn.conv2d_transpose(self.fus_pool3 ,W,deconv_shape, strides = [1,8,8,1], padding = 'SAME') 
             self.conv14 = conv 
             self.parameters+=[W]
             tf.summary.histogram("Final/W",W)   




    def biliner_W(input_, kernel_shape):
            
        width= kernel_shape[0]
        height= kernel_shape[1]
        f = np.ceil(width/2)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([width, height])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
                weights = np.zeros([width, height, 21, 21])
                for i in range(21):
                    weights[:, :, i, i] = bilinear
        return weights
