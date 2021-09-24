from tensorflow.keras.layers import Conv2D,Dropout,UpSampling2D,BatchNormalization,Activation,MaxPool2D,Conv2DTranspose,Concatenate,Input,concatenate
from tensorflow.keras.models import *
from numpy import shape
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.keras.layers.pooling import MaxPooling2D

def unet(pretrained_weights = None,input_size = (256,256,1)): 

    inputs = Input(input_size)
    
    #encoder 1
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) 
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

    #encoder 2
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1) 
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) 
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

    #encoder 3
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2) 
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) 
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 

    #encoder 4
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3) 
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4) 
    drop4 = Dropout(0.5)(conv4) 
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4) 

    # encoder 5
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4) 
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5) 
    drop5 = Dropout(0.5)(conv5) 

    #up sampling

    #decoder 1
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5)) 
    merge6 = concatenate([drop4,up6], axis = 3) 
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6) 
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6) 
    
    #decoder 2
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6)) 
    merge7 = concatenate([conv3,up7], axis = 3) 
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7) 
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7) 
    
    #decoder 3
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7)) 
    merge8 = concatenate([conv2,up8], axis = 3) 
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8) 
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8) 
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8)) 
    merge9 = concatenate([conv1,up9], axis = 3)    
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9) 
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    conv9 = Conv2D(3, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9) 
    # conv10 = Conv2D(3, 1, activation = 'relu')(conv9) 
    
    model = Model(inputs,conv9,name="unet") 
    
    model.compile(optimizer=Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy']) 
    # model.load_weights(pretrained_weights)
    return model
    