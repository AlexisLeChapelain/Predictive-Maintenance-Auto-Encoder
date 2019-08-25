import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
import json
from keras.models import model_from_json

from src.data.extract_and_reframe_serie import main_data_processing


class predict_anomaly:

    def __init__(self, folder_path):

        # Clear session in the beginning
        K.clear_session()

        # Path and name
        self.path= folder_path + "models/"
        self.name='autoencoder'
        self.model_autoencoder = None


    def restore_model(self):
        """
        Restore tf model, with weight and metadata. Initialize weight for prediction.
        """
        print("Start restoring model")
        # Load model
        json_file = open(self.path + "model_autoencoderBis.json", 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        print("Model is loaded")

        # Load weight
        self.model_autoencoder = model_from_json(loaded_model_json)
        self.model_autoencoder.load_weights(self.path + "weight_autoencoderBis.h5")
        print("Weigths are loaded")

        # load metadata
        with open(self.path + 'metadata_'+self.name+'.json', 'r') as json_file:
            metadata_autoencoder = json.load(json_file)
        json_file.close()
        metadata_autoencoder = eval(metadata_autoencoder)
        print("Metadata are loaded")

        # extract metadata from dictionary
        self.num_serie = metadata_autoencoder['num_serie']
        self.max_length=metadata_autoencoder["max_serie_length"]
        self.n_s_encoder = metadata_autoencoder['n_s_encoder']
        self.n_s_decoder = metadata_autoencoder['n_s_decoder']

        # initialize weights
        self.s0_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        self.c0_encoder = np.random.randn(self.num_serie, self.n_s_encoder)
        self.s0_decoder = np.random.randn(self.num_serie, self.n_s_decoder)
        self.c0_decoder = np.random.randn(self.num_serie, self.n_s_decoder)


    def load_data(self, dataname):
        """
        Load data
        :param dataname: name of the dataset
        :return data_dl: the reframe dataset on a numpy format
        :return max_length: the maximum length of the cycle contained in the
        """
        # Load data
        interim_data_folder = folder_path+ "data/interim/"
        data = pd.read_csv(interim_data_folder+"data_dl.csv", dtype={'dataValue': np.float64, 'pji': np.int64},
                           parse_dates=['sourceTimestamp_dtformat'], nrows=10000)
        data_dl, max_length, num_serie = main_data_processing(data, max_length=self.max_length)
        return data_dl, max_length


    def generate_prediction(self, dataname):
        # load data and maximum cycle length
        self.data_dl, self.max_length = self.load_data(dataname)
        # prediction
        self.predictions = self.model_autoencoder.predict([self.data_dl, self.s0_encoder, self.c0_encoder,
                                                           self.s0_decoder, self.c0_decoder])


    def visualize(self, id_serie=0):
        # visualization of a single serie (prediction vs data)
        predictions2 = self.predictions[id_serie,:,:]
        data_dl2 = self.data_dl[id_serie,:,0].reshape(-1,1)
        plt.plot(predictions2)
        plt.plot(data_dl2)
        plt.title('Real time series vs predicted')
        plt.ylabel('value')
        plt.xlabel('time')
        plt.legend(['Prediction', 'Real'], loc='upper right')
        plt.show()


if __name__ == '__main__':
    folder_path="/Users/az02234/Documents/Personnal_Git/PredictiveMaintenanceAutoEncoder/"
    predict_anomaly_cls = predict_anomaly(folder_path)
    predict_anomaly_cls.restore_model()
    predict_anomaly_cls.generate_prediction("data_dl.csv")
    predict_anomaly_cls.visualize(id_serie=0)