import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Bidirectional, Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation, Reshape
from keras.optimizers import Adam
from keras.models import Model
from keras.activations import softmax
import keras.backend as K

from src.data.extract_and_reframe_serie import main_data_processing


class time_serie_auto_encoder:

    def __init__(self, data, num_serie, max_serie_length, compression_size):

        # global variable for size
        self.data = data
        self.num_serie = num_serie
        self.compression_size = compression_size
        self.max_serie_length = max_serie_length

        # Defined shared layers of the encoder attention layer as global variables
        self.repeator_encoder = RepeatVector(self.max_serie_length)
        self.concatenator_encoder = Concatenate(axis=-1)
        self.densor1_encoder = Dense(10, activation="relu")
        self.densor2_encoder = Dense(1, activation="relu")
        self.activator_encoder = Activation(softmax, name='attention_weights_encoder')
        self.dotor_encoder = Dot(axes=1)

        # Defined shared layers  of the decoder attention layer as global variables
        self.repeator_decoder = RepeatVector(compression_size)
        self.concatenator_decoder = Concatenate(axis=-1)
        self.densor1_decoder = Dense(10, activation="relu")
        self.densor2_decoder = Dense(1, activation="relu")
        self.activator_decoder = Activation(softmax, name='attention_weights_decoder')
        self.dotor_decoder = Dot(axes=1)

        # global variable for encoder
        self.n_a_encoder = 16
        self.n_s_encoder = 32
        self.post_activation_LSTM_cell_encoder = LSTM(self.n_s_encoder, return_state=True)
        self.output_layer_encoder = Dense(2, activation='relu')

        # global variable for decoder
        self.n_a_decoder = 16
        self.n_s_decoder = 32
        self.post_activation_LSTM_cell_decoder = LSTM(self.n_s_decoder, return_state=True)
        self.output_layer_decoder = Dense(2, activation='relu')

        # store encoder_decoder instance
        self.encoder_decoder_instance = None
        self.fitting_history = None


    def fit(self, num_epoch=10, batch_size=256, validation_split=0.2, adam_config={'lr':0.01, 'beta_1':0.9, 'beta_2':0.999, 'decay':0.01}):
        """
        Instantiate an encoder_decoder and fit it
        :param num_epoch: number of epoch to fit the model
        :param batch_size: batch size
        :param adam_config: configuration of the ADAM optimizer. See KERAS documentation
        """

        self.encoder_decoder_instance = self.encoder_decoder()

        opt = Adam(**adam_config)
        self.encoder_decoder_instance.compile(opt, loss='mean_squared_error')

        s0_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        c0_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        s0_decoder = np.random.randn(self.num_serie, self.n_s_decoder)
        c0_decoder = np.random.randn(self.num_serie, self.n_s_decoder)

        target = self.data[:, :, :].reshape((self.num_serie, self.max_serie_length, 2))
        self.fitting_history = self.encoder_decoder_instance.fit([self.data, s0_encoder, c0_encoder, s0_decoder, c0_decoder],
                                                                 target, validation_split=validation_split, epochs=num_epoch,
                                                                 batch_size=batch_size)


    def save_model(self, path='/Users/az02234/Documents/Projets_Renault/PredictiveMaintenance/PredictiveMaintenanceAutoEncoder/models/', name='autoencoder'):
        """
        Save model in hdf5 / JSON
        """
        adress = path + name + '.h5'
        self.encoder_decoder_instance.save(adress)
        print("Model saved in {}".format(adress))

        self.encoder_decoder_instance.save_weights(path + 'weight_' + name + '.h5')
        model_json = self.encoder_decoder_instance.to_json()
        with open(path + 'model_'+name+'.json', "w") as json_file:
            json_file.write(model_json)
        json_file.close()

        metadata_json = {'num_serie': self.num_serie, 'compression_size': self.compression_size,
                         'max_serie_length': self.max_serie_length, 'n_s_encoder': self.n_s_encoder,
                         'n_s_decoder': self.n_s_decoder}
        metadata_json_str = json.dumps(metadata_json)
        with open(path + 'metadata_'+name+'.json', "w") as json_file:
            json.dump(metadata_json_str, json_file)
        json_file.close()


    def visualize_fit_history(self):
        """
        summarize history for loss
        """
        plt.plot(self.fitting_history.history['loss'])
        plt.plot(self.fitting_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


    def encoder_decoder(self):
        """
        Autoencoder for time series, with an attention mechanism.
        :return model: a keras model
        """
        Tx = self.max_serie_length

        # Define encoder input
        X = Input(shape=(Tx, 2,))
        s0_encoder = Input(shape=(self.n_s_encoder,), name='s0')
        c0_encoder = Input(shape=(self.n_s_encoder,), name='c0')
        s_encoder = s0_encoder
        c_encoder = c0_encoder
        #s_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        #c_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        #s_encoder = K.random_normal_variable(shape=(self.num_serie,self.n_s_encoder), mean=0, scale=0.1)
        #c_encoder = K.random_normal_variable(shape=(self.num_serie,self.n_s_encoder), mean=0, scale=0.1)

        # Initialize empty list of compressed inputs
        outputs = []

        # Step 1: Define the pre-attention Bi-LSTM.
        a = Bidirectional(LSTM(self.n_a_encoder, return_sequences=True))(X)

        # Step 2: Iterate for compression_size steps
        for t in range(self.compression_size):
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t
            context = self.one_step_attention_encoder(a, s_encoder)

            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            s_encoder, _, c_encoder = self.post_activation_LSTM_cell_encoder(context, initial_state=[s_encoder, c_encoder])

            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM
            out = self.output_layer_encoder(s_encoder)
            out = Reshape((1, 2))(out)

            # Step 2.D: Append "out" to the "outputs" list
            outputs.append(out)

        # Step 3: oncatenate and reshape the output list to get a compressed serie
        compressed_output = Concatenate(axis=-1)(outputs)
        compressed_output = Reshape((self.compression_size, 2))(compressed_output)

        # Define decoder input
        s0_decoder = Input(shape=(self.n_s_encoder,), name='s0_d')
        c0_decoder = Input(shape=(self.n_s_encoder,), name='c0_d')
        s_decoder = s0_decoder
        c_decoder = c0_decoder
        #s_decoder = np.random.randn(self.num_serie, self.n_s_decoder)
        #c_decoder = np.random.randn(self.num_serie, self.n_s_decoder)
        #s_decoder = K.random_normal_variable(shape=(self.num_serie, self.n_s_encoder), mean=0, scale=0.1)
        #c_decoder = K.random_normal_variable(shape=(self.num_serie, self.n_s_encoder), mean=0, scale=0.1)

        # Below: same three steps as in the encoder part
        outputs = []
        a_decoder = Bidirectional(LSTM(self.n_a_decoder, return_sequences=True))(compressed_output)

        for t in range(self.max_serie_length):
            context = self.one_step_attention_decoder(a_decoder, s_decoder)
            s_decoder, _, c_decoder = self.post_activation_LSTM_cell_decoder(context, initial_state=[s_decoder, c_decoder])
            out = self.output_layer_decoder(s_decoder)
            outputs.append(out)

        final_output = Concatenate(axis=-1)(outputs)
        final_output = Reshape((self.max_serie_length, 2))(final_output)

        # Final Step : Create model instance
        model = Model(inputs=[X, s0_encoder, c0_encoder, s0_decoder, c0_decoder], outputs=final_output)

        return model


    def one_step_attention_encoder(self, a, s_prev):
        """
        Performs one step of attention for the encoder step: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        :param a: hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        :param s_prev: previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        :return context: context vector, input of the next (post-attetion) LSTM cell
        """

        s_prev = self.repeator_encoder(s_prev)
        concat = self.concatenator_encoder([a, s_prev])
        e = self.densor1_encoder(concat)
        e = self.densor2_encoder(e)
        alphas = self.activator_encoder(e)
        context = self.dotor_encoder([alphas, a])
        return context


    def one_step_attention_decoder(self, a, s_prev):
        """
        Performs one step of attention for the decoder step: Outputs a context vector computed as a dot product of the attention weights
        "alphas" and the hidden states "a" of the Bi-LSTM.
        :param a: hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
        :param s_prev: previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
        :return context: context vector, input of the next (post-attetion) LSTM cell
        """

        s_prev = self.repeator_decoder(s_prev)
        concat = self.concatenator_decoder([a, s_prev])
        e = self.densor1_decoder(concat)
        e = self.densor2_decoder(e)
        alphas = self.activator_decoder(e)
        context = self.dotor_decoder([alphas, a])
        return context


if __name__ == '__main__':

    # get data and max_size
    os.chdir(
        '/users/az02234/Documents/Projets_Renault/PredictiveMaintenance/PredictiveMaintenanceAutoEncoder/data/interim/')

    data = pd.read_csv("data_dl.csv", dtype={'dataValue': np.float64, 'pji': np.int64},
                       parse_dates=['sourceTimestamp_dtformat'], nrows=500000)

    data_dl, max_length, num_serie = main_data_processing(data)
    data_dl = data_dl[:,:,[0,2]]

    time_serie_auto_encoder = time_serie_auto_encoder(data_dl, num_serie, max_length, 50)
    time_serie_auto_encoder.fit(num_epoch=20, batch_size=256)
    time_serie_auto_encoder.visualize_fit_history()
    time_serie_auto_encoder.save_model()

    print("Done!")

