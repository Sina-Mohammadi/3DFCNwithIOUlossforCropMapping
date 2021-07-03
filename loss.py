from tensorflow.keras import backend as K


def IOU(y_true, y_pred):
    
    epsilon   = K.constant(1e-22, dtype='float32')
    losses = []

    for j in range(3):

        y_true = y_true[...,j:j+1]
        y_true = K.batch_flatten(y_true)

        y_pred = y_pred[...,j:j+1] 
        y_pred = K.batch_flatten(y_pred)

        I = K.sum(y_pred * y_true , axis=-1)
        U = K.sum(y_pred + y_true -(y_pred * y_true) , axis=-1) 
        IOU = I / (U + epsilon)
        Mean_IOU = K.mean(IOU)
        IOU_Loss = 1 - Mean_IOU
        losses.append(IOU_Loss)


    return K.sum(losses)
