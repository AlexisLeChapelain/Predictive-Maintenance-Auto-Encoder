import warnings
warnings.filterwarnings("ignore")
import math
import os

import pandas as pd
import numpy as np


def main_data_processing(data, max_length='auto'):
    if max_length=='auto':
        max_length, num_serie = compute_max_sequence_length(data)

    data["micro_second"] = data.sourceTimestamp_dtformat.dt.microsecond \
                           + data.sourceTimestamp_dtformat.dt.second * 1000 \
                           + data.sourceTimestamp_dtformat.dt.minute * 1000 * 60 \
                           + data.sourceTimestamp_dtformat.dt.hour * 1000 * 60 * 60 \
                           + data.sourceTimestamp_dtformat.dt.day * 1000 * 60 * 60 * 24

    data = data.groupby("pji").apply(normalise)
    data = data.groupby("pji").apply(compute_lag)
    data = data.reset_index(drop=True)
    data = data.fillna(0)
    data = padding(data, max_length)
    return data, max_length


def compute_max_sequence_length(dataframe):
    lengths = dataframe.groupby("pji").count()
    num_serie = lengths.shape[0]
    max_length = lengths.max()[0]
    return max_length, num_serie


def normalise(dataframe):
    dataframe["micro_second"] = dataframe["micro_second"]/1000
    dataframe["normalized_time"] = (dataframe["micro_second"] - dataframe["micro_second"].min()) / dataframe["micro_second"].max()
    return dataframe


def compute_lag(dataframe):
    dataframe.sort_values(by=["normalized_time"], inplace=True)
    dataframe['interval'] = dataframe["micro_second"] - dataframe["micro_second"].shift()
    return dataframe


def padding(data, max_length):
    data_for_dl = []
    for vehicle_id in data.pji.drop_duplicates():
        # select vehicle data
        selector = (data.pji == vehicle_id)
        vehicle_data = data.loc[selector, ['dataValue', 'normalized_time', 'interval']]

        # compute length of the padding
        padding_length = max_length - vehicle_data.shape[0]
        if padding_length < 0:
            pass
        else:
            # add padding
            pad = (np.repeat(np.array([0, 1, 0]).reshape(-1, 1), padding_length, axis=1)).T
            vehicle_data = np.vstack((vehicle_data, pad))

            # Append to final array
            data_for_dl.append(list(vehicle_data))
    data_for_dl = np.array(data_for_dl)
    return data_for_dl


if __name__ == '__main__':
    os.chdir(
        '/users/az02234/Documents/Projets_Renault/PredictiveMaintenance/PredictiveMaintenanceAutoEncoder/data/interim/')

    data = pd.read_csv("data_dl.csv", dtype={'dataValue': np.float64, 'pji': np.int64},
                       parse_dates=['sourceTimestamp_dtformat'], nrows=500000)

    data, max_length, num_serie = main_data_processing(data)
    print("\n", max_length, "\n", num_serie)

    print("Done")
