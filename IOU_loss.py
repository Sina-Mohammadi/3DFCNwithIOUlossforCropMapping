from tensorflow.keras import backend as K


def IOU(y_true, y_pred):
    
    epsilon   = K.constant(1e-22, dtype='float32')
    losses = []

    for j in range(3):

        y_true1 = y_true[...,j:j+1]
        y_true1  = K.batch_flatten(y_true1)

        y_pred1 = y_pred[...,j:j+1] 
        y_pred1  = K.batch_flatten(y_pred1)

        I = K.sum(y_pred1 * y_true1 , axis=-1)
        U        = K.sum(y_pred1+y_true1 -(y_pred1 * y_true1) , axis=-1) 
        IOU   = I / (U + eps)
        Mean_IOU   = K.mean(IOU)
        IOU_Loss1 =1-Mean_IOU
        losses.append(IOU_Loss1)


    return K.sum(losses)
