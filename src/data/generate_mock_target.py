import warnings
warnings.filterwarnings("ignore")
import math
import os

import pandas as pd
import numpy as np

# Aim : build a a mock target for a neural network to predict
# The target is made of a smaller serie constituted of statistics from the original serie (average, variance, etc)


## To be optimized : first build a complete vector of block id and then concatenate it rather than doing multiple
## select within the df
def build_block(exemple, vehicle_id, num_block=10):
    exemple.sort_values(by=['pji', 'normalized_time'], inplace=True)
    selector = (exemple.pji == vehicle_id)
    vehicle_data = exemple[selector]
    length = vehicle_data.shape[0]
    vehicle_data["block"] = 0
    index_block = vehicle_data.columns.get_loc("block")
    for i in range(num_block):
        if i < 9:
            vehicle_data.iloc[math.floor(length/num_block) *(i): math.floor(length/num_block) *(i+1),index_block] = i
        else:
            vehicle_data.iloc[math.floor(length/num_block) *(i): length,index_block] = i
    return vehicle_data


def extract_stat_by_block(vehicle_data):
    mean = np.around(vehicle_data.groupby("block")["dataValue"].mean())
    std = np.around(vehicle_data.groupby("block")["dataValue"].std())
    median = np.around(vehicle_data.groupby("block")["dataValue"].median())
    first_quartile = np.around(vehicle_data.groupby("block")["dataValue"].quantile(0.25))
    third_quartile = np.around(vehicle_data.groupby("block")["dataValue"].quantile(0.75))

    target = pd.concat([mean, std, median, first_quartile, third_quartile], axis=0)
    target.reset_index(drop=True, inplace=True)
    #target = np.array(target).reshape(-1,1)
    target = np.array(target)
    return target


def build_target(exemple, num_block=10):
    target_list = []
    shuffled_index = np.array(range(num_block*5))
    np.random.shuffle(shuffled_index)
    for vehicle_id in exemple.pji.drop_duplicates():
        vehicle_data = build_block(exemple, vehicle_id, num_block=num_block)
        target = extract_stat_by_block(vehicle_data)
        target = target[shuffled_index.argsort()]
        target_list.append(list(target))
    target_list = np.array(target_list)
    return target_list


if __name__ == '__main__':
    #target_list = build_target(exemple, vehicle_id, num_block=10)
    #target_list.shape

    print("Done")