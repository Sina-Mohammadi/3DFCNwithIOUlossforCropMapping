import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
import argparse
import tables
import random
from numpy.random import seed
from model import FCN_3D
from loss import IOU

seed(1)
tensorflow.random.set_seed(1)



def args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./data/')
    parser.add_argument("--save_dir", type=str, default='./save/')
    parser.add_argument("--loss_function", type=str, default='IOU',choices=['IOU', 'Cross-entropy']) 
    parser.add_argument("--validation_fold", type=int, default=5)    
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=0.005)
    parser.add_argument("--epochs", type=int, default=150)

    return parser.parse_args()



if __name__ == "__main__":


    cfg = args()
    
    hdf5_file = tables.open_file(cfg.data_dir+'/data201720162015_1700_SiteA_128.hdf5', mode='r+')
    
    folds = [0,118,236,354,471,588]
    i = np.arange(588)
    j = np.arange(folds[cfg.validation_fold-1], folds[cfg.validation_fold])
    k = np.delete(i,j)
    train_images = hdf5_file.root.data[k, :, :, :, :]
    train_labels = hdf5_file.root.truth[k, :, :]
    
    val_images = hdf5_file.root.data[j, :, :, :, :]
    val_lables = hdf5_file.root.truth[j, :, :]
    
    
    
    
    def batch_generator(data, batch_size):
        mm = 0
        zz = np.arange(len(data[0]))
        random.shuffle(zz)
        
        while True:
            mm += batch_size
            if mm > len(data[0]):
                mini_batch_indices = zz[mm-batch_size:len(data[0])]
                r = mm-len(data[0])
                mm = 0
                zz = np.arange(len(data[0]))
                random.shuffle(zz)
                mm += r
                mini_batch_indices = np.concatenate([mini_batch_indices, zz[0:mm]])
            else:    
                mini_batch_indices = zz[mm-batch_size:mm]
            imgs = []
            lbls = []
    
    
            for t in mini_batch_indices:
                imgs.append(data[0][t])
                lbls.append(to_categorical(data[1][t],3))
    
            imgs_batch = np.array(imgs)
            lbls_batch = np.array(lbls)
    
            yield imgs_batch, lbls_batch
    
    
    train_batches = batch_generator((train_images, train_labels), batch_size = cfg.batch_size)
    val_batches   = batch_generator((val_images, val_lables), batch_size = cfg.batch_size)
    

    
    if cfg.loss_function == 'IOU':
        
        loss_function = IOU
        
    elif cfg.loss_function == 'Cross-entropy':
        
        loss_function = 'categorical_crossentropy'
        
    else:
        
      raise ValueError(" The name of the loss_function must be one of the following options: 'IOU', 'Cross_entropy' ")
        

    
    
    model = FCN_3D()
    model.compile(optimizer = SGD(lr = cfg.learning_rate, momentum = 0.9), loss = loss_function, metrics = ["accuracy"])
    



    class custom_callback(tensorflow.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            current = logs.get("accuracy")
            if np.greater(current, 0.97):
                self.model.stop_training = True



    check_point1 = callbacks.ModelCheckpoint(cfg.save_dir+'/m.hdf5', monitor = 'accuracy', verbose=1)
    check_point2 = callbacks.ModelCheckpoint(cfg.save_dir+'/m-{accuracy:.4f}-{epoch:02d}-{val_accuracy:.4f}.hdf5',
                                             monitor = 'val_accuracy', verbose = 1, save_best_only = True , mode = 'max')


    model.fit_generator(train_batches, steps_per_epoch = int(np.ceil(len(train_labels)/cfg.batch_size)), epochs = cfg.epochs, verbose = 1,
                        callbacks = [custom_callback(), check_point1], shuffle = True)
    
    model.fit_generator(train_batches, steps_per_epoch = int(np.ceil(len(train_labels)/cfg.batch_size)), epochs = cfg.epochs, verbose = 1,
                        callbacks = [check_point1, check_point2], validation_data = val_batches, validation_steps = int(np.ceil(len(val_lables)/cfg.batch_size)), shuffle = True)
