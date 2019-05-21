import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import math
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda, Reshape
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
from keras.activations import softmax
import keras.backend as K

from src.data.extract_and_reframe_serie import main_data_processing
from src.data.generate_mock_target import build_target


def one_step_attention_encoder(a, s_prev):
    """
    Performs one step of attention for the encoder step: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    :param a: hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    :param s_prev: previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    :return context: context vector, input of the next (post-attetion) LSTM cell
    """

    s_prev = repeator_encoder(s_prev)
    concat = concatenator_encoder([a, s_prev])
    e = densor1_encoder(concat)
    e = densor2_encoder(e)
    alphas = activator_encoder(e)
    context = dotor_encoder([alphas, a])
    return context


def one_step_attention_decoder(a, s_prev):
    """
    Performs one step of attention for the decoder step: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    :param a: hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    :param s_prev: previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    :return context: context vector, input of the next (post-attetion) LSTM cell
    """

    s_prev = repeator_decoder(s_prev)
    concat = concatenator_decoder([a, s_prev])
    e = densor1_decoder(concat)
    e = densor2_decoder(e)
    alphas = activator_decoder(e)
    context = dotor_decoder([alphas, a])
    return context


def encoder_decoder(max_length=300):
    """
    Encoder / decoder
    :return: a keras model
    """
    Tx = max_length

    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, 3,))
    s0_encoder = Input(shape=(n_s_encoder,), name='s0')
    c0_encoder = Input(shape=(n_s_encoder,), name='c0')
    s_encoder = s0_encoder
    c_encoder = c0_encoder

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a_encoder, return_sequences=True))(X)

    # Step 2: Iterate for Ty steps
    for t in range(Ty):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention_encoder(a, s_encoder)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s_encoder, _, c_encoder = post_activation_LSTM_cell_encoder(context, initial_state=[s_encoder, c_encoder])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer_encoder(s_encoder)
        out = Reshape((1,2))(out)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # compressed serie
    compressed_output = Concatenate(axis=-1)(outputs)
    compressed_output = Reshape((Ty, 2))(compressed_output)

    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    s0_decoder = Input(shape=(n_s_encoder,), name='s0_d')
    c0_decoder = Input(shape=(n_s_encoder,), name='c0_d')
    s_decoder = s0_decoder
    c_decoder = c0_decoder

    # Initialize empty list of outputs
    outputs = []

    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a_decoder = Bidirectional(LSTM(n_a_decoder, return_sequences=True))(compressed_output)

    # Step 2: Iterate for Tx steps
    for t in range(Tx):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention_decoder(a_decoder, s_decoder)

        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s_decoder, _, c_decoder = post_activation_LSTM_cell_decoder(context, initial_state=[s_decoder, c_decoder])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer_decoder(s_decoder)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    final_output = Concatenate(axis=-1)(outputs)
    final_output = Reshape((Tx, 1))(final_output)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X, s0_encoder, c0_encoder, s0_decoder, c0_decoder], outputs=final_output)

    return model



if __name__ == '__main__':

    # get data and max_size
    os.chdir(
        '/users/az02234/Documents/Projets_Renault/PredictiveMaintenance/PredictiveMaintenanceAutoEncoder/data/interim/')

    data = pd.read_csv("data_dl.csv", dtype={'dataValue': np.float64, 'pji': np.int64},
                       parse_dates=['sourceTimestamp_dtformat'], nrows=100000)

    data_dl, max_length, num_serie = main_data_processing(data)

    # global variable for size
    m = num_serie
    Ty = 50
    Tx = max_length

    # Defined shared layers of the encoder attention layer as global variables
    repeator_encoder = RepeatVector(Tx)
    concatenator_encoder = Concatenate(axis=-1)
    densor1_encoder = Dense(10, activation = "relu")
    densor2_encoder = Dense(1, activation = "relu")
    activator_encoder = Activation(softmax, name='attention_weights_encoder')
    dotor_encoder = Dot(axes = 1)

    # Defined shared layers  of the decoder attention layer as global variables
    repeator_decoder = RepeatVector(Ty)
    concatenator_decoder = Concatenate(axis=-1)
    densor1_decoder = Dense(10, activation = "relu")
    densor2_decoder = Dense(1, activation = "relu")
    activator_decoder = Activation(softmax, name='attention_weights_decoder')
    dotor_decoder = Dot(axes = 1)


    # global variable for encoder
    n_a_encoder = 16
    n_s_encoder = 32
    post_activation_LSTM_cell_encoder = LSTM(n_s_encoder, return_state = True)
    output_layer_encoder = Dense(2, activation='relu')

    # global variable for decoder
    n_a_decoder = 16
    n_s_decoder = 32
    post_activation_LSTM_cell_decoder = LSTM(n_s_decoder, return_state = True)
    output_layer_decoder = Dense(1, activation='relu')


    encoder_decoder = encoder_decoder(max_length=max_length)

    opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
    encoder_decoder.compile(opt, loss='mean_squared_error')

    s0_encoder = np.random.randn(m, n_s_encoder)
    c0_encoder = np.random.randn(m, n_s_encoder)
    s0_decoder = np.random.randn(m, n_s_decoder)
    c0_decoder = np.random.randn(m, n_s_decoder)

    target = data_dl[:,:,0].reshape((m, max_length, 1))
    encoder_decoder.fit([data_dl, s0_encoder, c0_encoder, s0_decoder, c0_decoder], target, epochs=200, batch_size=64)

    print("Done")



