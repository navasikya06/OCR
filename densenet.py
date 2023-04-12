from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ZeroPadding2D, SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Input, Flatten, UpSampling2D, Dense
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Concatenate, Add, Reshape, Permute, Multiply
from tensorflow.keras.layers import BatchNormalization, TimeDistributed
from tensorflow.keras.layers import Dropout, LeakyReLU, ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


def depthwise_separable_conv_block(inputs, pointwise_conv_filters, alpha, filter_size=(3, 3),
                                   depth_multiplier=1, strides=(1, 1),
                                   block_name='depthwise_separable_conv_block'):
    with K.name_scope(block_name):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        pointwise_conv_filters = int(pointwise_conv_filters * alpha)

        if strides == (1, 1):
            x = inputs
        else:
            x = ZeroPadding2D(((0, 1), (0, 1)))(inputs)

        x = DepthwiseConv2D(filter_size,
                            padding='same' if strides == (1, 1) else 'valid',
                            depth_multiplier=depth_multiplier,
                            strides=strides,
                            use_bias=False)(x)

        x = SeparableConv2D(pointwise_conv_filters, (1, 1),
                            padding='same' if strides == (1, 1) else 'valid',
                            depth_multiplier=depth_multiplier,
                            strides=(1, 1),
                            use_bias=False)(x)

        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(6.)(x)
        x = Dropout(.2)(x)
    return x


def conv_block(inputs, growth_rate, dropout_rate, weight_decay, block_name='conv_block'):
    with K.name_scope(block_name):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        x = BatchNormalization(axis=channel_axis)(inputs)
        x = ReLU(6.)(x)
        x = Conv2D(growth_rate, (3, 3), kernel_initializer='he_normal', padding='same')(x)
        x = depthwise_separable_conv_block(x, pointwise_conv_filters=growth_rate, alpha=1, filter_size=(3, 3))
        if (dropout_rate):
            x = Dropout(dropout_rate)(x)
    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate, weight_decay, block_name='dense_block'):
    with K.name_scope(block_name):
        for i in range(nb_layers):
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
            conv_block_name = f'{block_name}-conv_block-{i}'
            cb = conv_block(x, growth_rate, droput_rate, weight_decay, conv_block_name)
            x = Concatenate(axis=channel_axis)([x, cb])
            nb_filter += growth_rate
    return x, nb_filter


def self_attention(inputs, nb_filter, block_name='self_attention'):
    with K.name_scope(block_name):
        f = Conv2DTranspose(filters=nb_filter, kernel_size=1, strides=1, padding="same", data_format="channels_last")(
            inputs)
        g = Conv2D(filters=nb_filter, kernel_size=1, strides=1, padding="same")(inputs)
        h = Conv2D(filters=nb_filter, kernel_size=1, strides=1, padding="same")(inputs)
        sa = Multiply()([Multiply()([f, g]), h])
        sa = Conv2D(filters=nb_filter, kernel_size=1, strides=1, padding="same")(sa)
        sa = Multiply()([inputs, sa])
    return sa


def fast_dense_block(x, nb_layers, nb_filter, growth_rate, droput_rate, weight_decay, block_name='fast_dense_block'):
    with K.name_scope(block_name):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        fconv_block_name = f'{block_name}-fconv_block-{0}'
        x0 = conv_block(x, growth_rate, droput_rate, weight_decay, fconv_block_name)
        skip_connections = []
        for i in range(nb_layers - 1):
            fconv_block_name = f'{block_name}-fconv_block-{i + 1}'
            x1 = conv_block(x0, growth_rate, droput_rate, weight_decay, fconv_block_name)
            skip_connections.append(x1)
            for sc in skip_connections:
                x0 = Add()([x0, sc])
        x = Concatenate(axis=channel_axis)([x, x0])
        nb_filter += growth_rate
    return x, nb_filter


def transition_block(inputs, nb_filter, dropout_rate, pooltype, weight_decay, block_name='transition_block'):
    with K.name_scope(block_name):
        conv = Conv2D(nb_filter, (1, 1), kernel_initializer='he_normal', padding='same', use_bias=False,
                      kernel_regularizer=l2(weight_decay))(inputs)
        if (dropout_rate):
            conv = Dropout(dropout_rate)(conv)
        if (pooltype == 1):
            x = AveragePooling2D((2, 2), strides=(2, 2))(conv)
        elif (pooltype == 2):
            x = MaxPooling2D()(conv)
        elif (pooltype == 3):
            x = AveragePooling2D((2, 2), strides=(2, 1))(conv)
        elif (pooltype == 4):
            x = Conv2D(nb_filter, (2, 2), strides=(2, 2), kernel_initializer='he_normal', padding='same')(conv)
    return x, nb_filter


def deconv(inputs, nb_filter, dropout_rate, pooltype, weight_decay, block_name='deconv'):
    with K.name_scope(block_name):
        conv = Conv2D(nb_filter, (3, 3), kernel_initializer='he_normal', padding='same')(inputs)
        # conv = depthwise_separable_conv_block(inputs,
        #                                     pointwise_conv_filters= nb_filter,
        #                                     alpha= 1,
        #                                     filter_size= (3, 3))
        if (dropout_rate):
            conv = Dropout(dropout_rate)(conv)
        if (pooltype == 1):
            pool = UpSampling2D(size=(2, 2), interpolation='bilinear')(conv)
        elif (pooltype == 2):
            pool = Conv2DTranspose(nb_filter, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv)
    return pool, nb_filter


def db_encoding_module(x, k, _nb_filter, _dropout_rate, _weight_decay, block_name='bd_encoding_module'):
    with K.name_scope(block_name):
        # conv 64 5*5 s=2
        print('input', x)
        x = Conv2D(_nb_filter, (5, 5),
                   strides=(2, 2),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(_weight_decay))(x)

        # x = depthwise_separable_conv_block(x,
        #                                     pointwise_conv_filters= _nb_filter,
        #                                     alpha= 1,
        #                                     filter_size= (5, 5),
        #                                     strides=(2, 2))

        print('conv', x)
        # 64 + 8*8 = 128
        x, _nb_filter = fast_dense_block(x, k, _nb_filter, 64, None, _weight_decay, 'dense_block-0')
        print(x, _nb_filter)
        # 128
        x, _nb_filter = transition_block(x, 128, _dropout_rate, 4, _weight_decay, 'transition_block-0')
        print(x, _nb_filter)
        # 128 + 8*8 = 192
        x, _nb_filter = fast_dense_block(x, k, _nb_filter, 64, None, _weight_decay, 'dense_block-1')
        print(x, _nb_filter)
        # 192 -> 128
        x, _nb_filter = transition_block(x, 128, _dropout_rate, 4, _weight_decay, 'transition_block-1')
        print(x, _nb_filter)
        # 128 + 8*8 = 192
        x, _nb_filter = fast_dense_block(x, k, _nb_filter, 64, None, _weight_decay, 'dense_block-2')
    return x, _nb_filter


def up_sampling_block(x, k, _nb_filter, _dropout_rate, _weight_decay, block_name='up_sampling_block'):
    # 4x16x128
    with K.name_scope(block_name):
        ux, _nb_filter = deconv(x, 128, _dropout_rate, 2, _weight_decay, 'deconv-0')
        print(ux, _nb_filter)
        # 64 + 8*8 = 128
        x, _nb_filter = fast_dense_block(ux, k, _nb_filter, 64, None, _weight_decay, 'dense_block-0')
        print(x, _nb_filter)
        # 128
        x, _nb_filter = transition_block(x, 128, _dropout_rate, 4, _weight_decay, 'transition_block-0')
        print(x, _nb_filter)
        # 128 + 8*8 = 192
        x, _nb_filter = fast_dense_block(ux, k, _nb_filter, 64, None, _weight_decay, 'dense_block-0')
        print(x, _nb_filter)

        # x = depthwise_separable_conv_block(x,pointwise_conv_filters= _nb_filter,
        #                                     alpha= 1,
        #                                     filter_size= (3, 3),
        #                                     strides=(1, 1))
        x = Conv2D(_nb_filter, (5, 5),
                   strides=(2, 2),
                   kernel_initializer='he_normal',
                   padding='same',
                   use_bias=False,
                   kernel_regularizer=l2(_weight_decay))(x)
    return x, _nb_filter


def densenet_u(input_layer, nclass, block_name='densenet_u', _nb_filter=64):
    with K.name_scope(block_name):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        # ===========================================================================
        # FDB encoding module
        # ===========================================================================
        x, _nb_filter = db_encoding_module(input_layer, k=5,
                                           _nb_filter=_nb_filter,
                                           _dropout_rate=0.2,
                                           _weight_decay=1e-4,
                                           block_name='bd_encoding_module-0')
        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(6.)(x)
        # #===========================================================================
        # # Up-sampling block
        # #===========================================================================
        # x, _nb_filter = up_sampling_block(x, k=5,
        #                                 _nb_filter=_nb_filter,
        #                                 _dropout_rate=0.2,
        #                                 _weight_decay=1e-4,
        #                                 block_name ='up_sampling_block-0')
        # x = BatchNormalization(axis=channel_axis)(x)
        # x = ReLU(6.)(x)

        x = Permute((2, 1, 3), name='permute')(x)
        x = TimeDistributed(Flatten(), name='flatten')(x)
        # x = Flatten()(x)
        y_pred = Dense(nclass, name='out', activation='softmax')(x)
    return y_pred


def densenet(input_layer, nclass, block_name='fdensenet',
             _nb_filter=64,
             k=8,
             _dropout_rate=0.2,
             _weight_decay=1e-4):
    with K.name_scope(block_name):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        # conv 64 5*5 s=2
        # x = Conv2D(_nb_filter, (5, 5), strides=(2, 2), kernel_initializer='he_normal', padding='same')(input_layer)
        # print(x)
        x = depthwise_separable_conv_block(input_layer, pointwise_conv_filters=_nb_filter, alpha=1, filter_size=(5, 5),
                                           strides=(2, 2))
        # 64 + 8 = 72
        x, _nb_filter = dense_block(x, k, _nb_filter, 8, None, _weight_decay, 'dense_block-0')
        print(x, _nb_filter)
        # 128
        x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay, 'transition_block-0')
        print(x, _nb_filter)
        # 128 + 8 = 136
        x, _nb_filter = dense_block(x, k, _nb_filter, 8, None, _weight_decay, 'dense_block-1')
        print(x, _nb_filter)
        # 192 -> 128
        x, _nb_filter = transition_block(x, 128, _dropout_rate, 2, _weight_decay, 'transition_block-1')
        print(x, _nb_filter)
        # 128 + 8 = 136
        x, _nb_filter = dense_block(x, k, _nb_filter, 8, None, _weight_decay, 'dense_block-2')

        x = BatchNormalization(axis=channel_axis)(x)
        x = ReLU(6.)(x)

        # x = Permute((2, 1, 3), name='permute')(x)
        # x = TimeDistributed(Flatten(), name='flatten')(x)
        x = Flatten()(x)
        y_pred = Dense(nclass, name='out', activation='softmax')(x)

    return y_pred