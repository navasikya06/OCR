import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Activation, Flatten, TimeDistributed, Dropout, Reshape
from tensorflow.keras.layers import Concatenate, Permute, BatchNormalization, Lambda, LeakyReLU
from tensorflow.keras.optimizers import Adam
from densenet import *

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_lenth = args
    print(args)
    return K.ctc_batch_cost(labels, y_pred, input_length, label_lenth)

def cnn_ctc(training, input_shape, size_voc):
    inputs = Input(name='the_input', shape=input_shape, dtype='float32')
    y_pred = densenet_u(inputs, size_voc)
    labels = Input(name='the_labels', shape=[None], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    print(y_pred)
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    if training:

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
        model.compile(optimizer=adam, loss={'ctc': lambda y_true, y_pred: y_pred}, metrics=['accuracy'])
        print("Model compiled successfully.")
        model.summary()
        return model
    else:
        return Model(inputs=inputs, outputs=y_pred)

"""
if __name__ == '__main__':
    model = cnn_ctc(training=True, input_shape = (32, 32, 3), size_voc=16)"""