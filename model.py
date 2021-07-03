from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization,Activation, Input, Conv3D , Concatenate , UpSampling3D, Lambda
K.set_image_data_format("channels_last")



def FCN_3D():
    
    inputlayer = Input(shape=(128,128,23,6))
    
    
    conv1 = Conv3D(32, kernel_size=(3,3,5), strides=(2,2,1), padding='same')(inputlayer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    
    
    
    conv2 = Conv3D(64, kernel_size=(3,3,5), strides=(2,2,1), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv3D(128, kernel_size=(3,3,5), strides=(2,2,1), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    
    
    conv4 = Conv3D(256, kernel_size=(3,3,5), strides=(2,2,1), padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv4U = UpSampling3D(size=(2, 2, 1))(conv4)
    conv4Uconv3 = Concatenate()([conv4U,conv3])
    
    
    conv5 = Conv3D(128, kernel_size=(3,3,5), strides=(1,1,1), padding='same')(conv4Uconv3)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)
    
    conv5U = UpSampling3D(size=(2, 2, 1))(conv5)
    conv5Uconv2 = Concatenate()([conv5U,conv2])
    
    conv6 = Conv3D(64, kernel_size=(3,3,5), strides=(1,1,1), padding='same')(conv5Uconv2)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    
    conv6U = UpSampling3D(size=(2, 2, 1))(conv6)
    conv6Uconv1 = Concatenate()([conv6U,conv1])
    
    
    conv7 = Conv3D(32, kernel_size=(3,3,5), strides=(1,1,1), padding='same')(conv6Uconv1)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)
    
    conv7U = UpSampling3D(size=(2, 2, 1))(conv7)
    
    conv8 = Conv3D(16, kernel_size=(3,3,5), strides=(1,1,1), padding='same')(conv7U)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)
    
    conv9 = Conv3D(3, kernel_size=(1,1,23), strides=(1,1,1))(conv8)
    squeezed = Lambda(lambda x: K.squeeze(x, 3))(conv9)
    
    out = Activation('softmax')(squeezed)
    
    
    
    model = Model(inputlayer,out)

    return model