# Load required python libraries

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import (Input, Dense, TimeDistributed, LSTM, GRU, Dropout, Concatenate,
                          Flatten, RepeatVector, RNN, Bidirectional, SimpleRNN)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import sample_data
import keras_util as ku

def encoder(model_input, layer, size, num_layers, drop_frac=0.0, output_size=None,
            bidirectional=False, **parsed_args):
    """Encoder module of autoencoder architecture.

    Can be used either as the encoding component of an autoencoder or as a standalone
    encoder, which takes (possibly irregularly-sampled) time series as inputs and produces
    a fixed-length vector as output.

    model_input: `keras.layers.Input`
        Input layer containing (y) or (dt, y) values
    layer: `keras.layers.Recurrent`
        Desired `keras` recurrent layer class
    size: int
        Number of units within each hidden layer
    num_layers: int
        Number of hidden layers
    drop_frac: float
        Dropout rate
    output_size: int, optional
        Size of encoding layer; defaults to `size`
    bidirectional: bool, optional
        Whether the bidirectional version of `layer` should be used; defaults to `False`
    """

    if output_size is None:
        output_size = size
    encode = model_input
    for i in range(num_layers):
        wrapper = Bidirectional if bidirectional else lambda x: x
        encode = wrapper(layer(size, name='encode_{}'.format(i),
                               return_sequences=(i < num_layers - 1)))(encode)
        if drop_frac > 0.0:
            encode = Dropout(drop_frac, name='drop_encode_{}'.format(i))(encode)
    encode = Dense(output_size, activation='linear', name='encoding')(encode)
    return encode

def decoder(encode, layer, n_step, size, num_layers, drop_frac=0.0, aux_input=None,
            bidirectional=False, **parsed_args):
    """Decoder module of autoencoder architecture.

    Can be used either as the decoding component of an autoencoder or as a standalone
    decoder, which takes a fixed-length input vector and generates a length-`n_step`
    time series as output.

    layer: `keras.layers.Recurrent`
        Desired `keras` recurrent layer class
    n_step: int
        Length of output time series
    size: int
        Number of units within each hidden layer
    num_layers: int
        Number of hidden layers
    drop_frac: float
        Dropout rate
    aux_input: `keras.layers.Input`, optional
        Input layer containing `dt` values; if `None` then the sequence is assumed to be
        evenly-sampled
    bidirectional: bool, optional
        Whether the bidirectional version of `layer` should be used; defaults to `False`
    """
    decode = RepeatVector(n_step, name='repeat')(encode)
    if aux_input is not None:
        decode = Concatenate()([aux_input, decode])

    for i in range(num_layers):
        if drop_frac > 0.0 and i > 0:  # skip these for first layer for symmetry
            decode = Dropout(drop_frac, name='drop_decode_{}'.format(i))(decode)
        wrapper = Bidirectional if bidirectional else lambda x: x
        decode = wrapper(layer(size, name='decode_{}'.format(i),
                               return_sequences=True))(decode)

    decode = TimeDistributed(Dense(1, activation='linear'), name='time_dist')(decode)
    return decode

def main(args=None):
    """Generate random periodic time series and train an autoencoder model.
    
    args: dict
        Dictionary of values to override default values in `keras_util.parse_model_args`;
        can also be passed via command line. See `parse_model_args` for full list of
        possible arguments.
    """
    args = ku.parse_model_args(args)

    train = np.arange(args.N_train); test = np.arange(args.N_test) + args.N_train
    npzfile = np.load('/Users/juan/L3/ml_prototype/training_data_rnn.npy.npz',allow_pickle=True)
    X1 = npzfile['arr_0']
    Y = npzfile['arr_1']
    
    
    Y[np.isnan(Y)] = 0.5
    print(Y)
    
    X = tf.keras.preprocessing.sequence.pad_sequences(
    X1, maxlen=None, dtype="float", padding="post", truncating="pre", value=0.0)


    model_type_dict = {'gru': GRU, 'lstm': LSTM, 'vanilla': SimpleRNN}

    main_input = Input(shape=(X.shape[1], X.shape[-1]), name='main_input')
    if args.even:
        model_input = main_input
        aux_input = None
    else:
        aux_input = Input(shape=(X.shape[1], X.shape[-1] - 1), name='aux_input')
        model_input = [main_input, aux_input]

    encode = encoder(main_input, layer=model_type_dict[args.model_type],
                     output_size=args.embedding, **vars(args))
    decode = decoder(encode, num_layers=args.decode_layers if args.decode_layers
                                                           else args.num_layers,
                     layer=model_type_dict[args.decode_type if args.decode_type
                                           else args.model_type],
                     n_step=X.shape[1], aux_input=aux_input,
                     **{k: v for k, v in vars(args).items() if k != 'num_layers'})
    model = Model(model_input, decode)
    model.compile(optimizer="Adam", loss="mse" )

    #run = ku.get_run_id(**vars(args))
 
    #if args.even:
    #history = ku.train_and_log(X[train], Y[train], run, model, **vars(args))
    #history = ku.train_and_log(X[train], Y[train], run, model, nb_epoch, batch_size, lr, loss, sim_type, metrics=[],
    #              sample_weight=None, no_train=False, patience=20, finetune_rate=None,
    #              validation_split=0.2, validation_data=None, gpu_frac=None,
    #              noisify=False, errors=None, pretrain_weights=None)
    history = model.fit(X, Y, batch_size=64, validation_split=0.2, epochs=1)
    
    #else:
    #    sample_weight = (X[train, :, -1] != -1)
    #    history = ku.train_and_log({'main_input': X[train], 'aux_input': X[train, :, 0:1]},
    #                               X_raw[train, :, 1:2], run, model,
    #                               sample_weight=sample_weight, **vars(args))
    return X, Y, model, args

if __name__ == '__main__':
    X, Y, model, args = main()
