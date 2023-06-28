#!/usr/bin/env python
# coding: utf-8
# @Author   ：Zane
# @Mail     : zanezii@foxmail.com
# @Date     ：2023/5/26 11:20
# @File     ：models.py
# @Description :
#               cGan with attention U-net
#               cGan with U-net
#               U-net
#               ConvLSTM
#               pySTEPS: https://github.com/pySTEPS/pysteps/tree/master/pysteps
import tensorflow as tf
from keras.optimizers import adam_v2 as adam
from keras.models import Model
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.convolutional import Conv3D
from keras.layers import LayerNormalization,BatchNormalization,concatenate,multiply,ReLU,add, Reshape
from keras.layers import LeakyReLU,Input,Conv2D,Activation,MaxPooling2D,Dropout,UpSampling2D


def discriminator_model():
    in_inputs = Input(shape=(128, 128, 4), name='input')  # (None,128,128,4)
    tar_inputs = Input(shape=(128, 128, 1), name='target')  # (None,128,128,1)

    concat = concatenate([in_inputs, tar_inputs], axis=3)  # (None,128,128,5)

    conv1 = Conv2D(64, 4, strides=(2, 2), padding='same', kernel_initializer='he_normal')(concat)  # (None,64,64,64)
    bn1 = BatchNormalization()(conv1)
    act1 = LeakyReLU(alpha=0.2)(bn1)

    conv2 = Conv2D(128, 4, strides=(2, 2), padding='same', kernel_initializer='he_normal')(act1)  # (None,32,32,128)
    bn2 = BatchNormalization()(conv2)
    act2 = LeakyReLU(alpha=0.2)(bn2)

    conv4 = Conv2D(256, 4, strides=(1, 1), padding='same', kernel_initializer='he_normal')(act2)  # (None,32,32,256)
    bn4 = BatchNormalization()(conv4)
    act4 = LeakyReLU(alpha=0.2)(bn4)

    conv = Conv2D(1, 4, padding='same', kernel_initializer='he_normal')(act4)  # (None,32,32,1)
    outputs = Activation('sigmoid')(conv)

    model = Model([in_inputs, tar_inputs], outputs)

    opt = adam.Adam(learning_rate=0.001, beta_1=0.5)
    # opt=Adam(lr=0.001,beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def generator_model(model='cGan'):
    # input
    inputs = Input(shape=(128, 128, 4))  # (None,128,128,4)

    # Encoder: the contracting network
    conv1s = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(inputs)  # (None,128,128,128)
    bn1s = BatchNormalization()(conv1s)
    act1s = Activation('relu')(bn1s)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1s)  # (None,64,64,128)
    drop1 = Dropout(0.5)(pool1)

    conv2f = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop1)  # (None, 64,64,256)
    bn2f = BatchNormalization()(conv2f)
    act2f = Activation('relu')(bn2f)
    conv2s = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(act2f)  # (None, 64,64,512)
    bn2s = BatchNormalization()(conv2s)
    act2s = Activation('relu')(bn2s)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act2s)  # (None,32,32,512)
    drop2 = Dropout(0.5)(pool2)

    conv3f = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(drop2)  # (None,32,32,1024)
    bn3f = BatchNormalization()(conv3f)
    act3f = Activation('relu')(bn3f)
    drop3 = Dropout(0.5)(act3f)

    # Decoder: the expanding network
    smpl4 = UpSampling2D(size=(2, 2))(drop3)  # (None,64,64,1024)
    if model == 'cGan':
        up4 = concatenate([smpl4, act2s], axis=3)  # (None,32,32,512*3)
    else:
        att4 = AttentionGate(g=smpl4, X=act2s, channel=512, name='att4')
        up4 = concatenate([att4, smpl4], axis=3)
    conv4f = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up4)  # (None,32,32,512)
    bn4f = BatchNormalization()(conv4f)  # (None,64,64,512)
    act4f = Activation('relu')(bn4f)
    drop4f = Dropout(0.5)(act4f)
    conv4 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(
        drop4f)  # (None,64,64,256)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)

    smpl5 = UpSampling2D(size=(2, 2))(act4)  # (None,128,128,256)
    if model == 'cGan':
        up5 = concatenate([smpl5, act1s], axis=3)  # (None,128,128,128*3)
    else:
        att5 = AttentionGate(g=smpl5, X=act1s, channel=128, name='att5')
        up5 = concatenate([smpl5, att5], axis=3)
    conv5f = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up5)  # (None,128,128,128)
    bn5f = BatchNormalization()(conv5f)
    act5f = Activation('relu')(bn5f)
    drop5f = Dropout(0.5)(act5f)
    conv5s = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop5f)  # (None,128,128,64)
    bn5s = BatchNormalization()(conv5s)
    act5s = Activation('relu')(bn5s)
    drop5s = Dropout(0.5)(act5s)
    conv5 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(drop5s)  # (None,128,128,2)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)

    # output
    outputs = Conv2D(1, 1, activation='linear')(act5)  # (None,128,128,1)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def AttentionGate(X, g, channel,
                  activation='ReLU',
                  attention='add', name='att'):
    """
    Self-attention gate modified from Oktay et al. 2018.

    attention_gate(X, g, channel,  activation='ReLU', attention='add', name='att')

    Input
    ----------
        X: input tensor, i.e., key and value.
        g: gated tensor, i.e., query.
        channel: number of intermediate channel.
                 Oktay et al. (2018) did not specify (denoted as F_int).
                 intermediate channel is expected to be smaller than the input channel.
        activation: a nonlinear attnetion activation.
                    The `sigma_1` in Oktay et al. 2018. Default is 'ReLU'.
        attention: 'add' for additive attention; 'multiply' for multiplicative attention.
                   Oktay et al. 2018 applied additive attention.
        name: prefix of the created keras layers.

    Output
    ----------
        X_att: output tensor.

    """
    activation_func = eval(activation)
    attention_func = eval(attention)

    # mapping the input tensor to the intermediate channel
    theta_att = Conv2D(channel, 1, use_bias=True, name='{}_theta_x'.format(name))(X)

    # mapping the gate tensor
    phi_g = Conv2D(channel, 1, use_bias=True, name='{}_phi_g'.format(name))(g)

    # ----- attention learning ----- #
    query = attention_func([theta_att, phi_g], name='{}_add'.format(name))

    # nonlinear activation
    f = activation_func(name='{}_activation'.format(name))(query)

    # linear transformation
    psi_f = Conv2D(1, 1, use_bias=True, name='{}_psi_f'.format(name))(f)
    # ------------------------------ #

    # sigmoid activation as attention coefficients
    coef_att = Activation('sigmoid', name='{}_sigmoid'.format(name))(psi_f)

    # multiplicative attention masking
    X_att = multiply([X, coef_att], name='{}_masking'.format(name))

    return X_att


def Gan_model(generator_model, discriminator_model):
    discriminator_model.trainable = False
    inputs = Input(shape=(128, 128, 4))
    gen_out = generator_model(inputs)
    dis_out = discriminator_model([inputs, gen_out])

    model = Model(inputs, [dis_out, gen_out])

    # opt=Adam(lr=0.001,beta_1=0.5)
    opt = adam.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def convLSTM_model(input_shape=(128,128,4)):
    inputs = Input(input_shape)
    re0 = Reshape((1,128,128,4))(inputs)
    cov1 = ConvLSTM2D(filters=64,kernel_size=(3,3),padding='same',kernel_initializer='HeNormal',return_sequences=True)(re0)
    ln1 = LayerNormalization()(cov1)
    cov2 = ConvLSTM2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='HeNormal', return_sequences=True)(ln1)
    ln2 = LayerNormalization()(cov2)
    cov3 = ConvLSTM2D(filters=64, kernel_size=(3,3),padding='same', kernel_initializer='HeNormal', return_sequences=True)(ln2)
    ln3 = LayerNormalization()(cov3)
    cov4 = Conv3D(filters=1, kernel_size=(3,3,3),padding='same', activation='linear', data_format='channels_last')(ln3)
    outputs = Reshape((128,128,1))(cov4)

    model = Model(inputs=inputs, outputs=outputs)
    opt = adam.Adam(learning_rate=0.001, beta_1=0.5)
    model.compile(loss='mse',optimizer=opt)

    return model


def unet_model(input_shape=(128, 128, 4)):
    inputs = Input(input_shape)

    conv1s = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(inputs)
    bn1s = BatchNormalization()(conv1s)
    act1s = Activation('relu')(bn1s)
    pool1 = MaxPooling2D(pool_size=(2, 2))(act1s)
    drop1 = Dropout(0.5)(pool1)

    conv2f = Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(drop1)
    bn2f = BatchNormalization()(conv2f)
    act2f = Activation('relu')(bn2f)
    conv2s = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(act2f)
    bn2s = BatchNormalization()(conv2s)
    act2s = Activation('relu')(bn2s)
    pool2 = MaxPooling2D(pool_size=(2, 2))(act2s)
    drop2 = Dropout(0.5)(pool2)

    conv3f = Conv2D(1024, 3, padding='same', kernel_initializer='he_normal')(drop2)
    bn3f = BatchNormalization()(conv3f)
    act3f = Activation('relu')(bn3f)
    drop3 = Dropout(0.5)(act3f)

    up4 = concatenate([UpSampling2D(size=(2, 2))(drop3), act2s], axis=3)
    conv4f = Conv2D(512, 3, padding='same', kernel_initializer='he_normal')(up4)
    bn4f = BatchNormalization()(conv4f)
    act4f = Activation('relu')(bn4f)
    drop4f = Dropout(0.5)(act4f)
    conv4 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(drop4f)
    bn4 = BatchNormalization()(conv4)
    act4 = Activation('relu')(bn4)

    up5 = concatenate([UpSampling2D(size=(2, 2))(act4), act1s], axis=3)
    conv5f = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(up5)
    bn5f = BatchNormalization()(conv5f)
    act5f = Activation('relu')(bn5f)
    drop5f = Dropout(0.5)(act5f)
    conv5s = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(drop5f)
    bn5s = BatchNormalization()(conv5s)
    act5s = Activation('relu')(bn5s)
    drop5s = Dropout(0.5)(act5s)
    conv5 = Conv2D(2, 3, padding='same', kernel_initializer='he_normal')(drop5s)
    bn5 = BatchNormalization()(conv5)
    act5 = Activation('relu')(bn5)

    outputs = Conv2D(1, 1, activation='linear')(act5)

    model = Model(inputs=inputs, outputs=outputs)

    opt = adam.Adam(learning_rate=0.001, beta_1=0.5)
    # model.compile(loss=['mean_absolute_error'], optimizer=opt, metrics=['mean_absolute_error'])
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model

