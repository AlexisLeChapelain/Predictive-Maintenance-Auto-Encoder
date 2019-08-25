# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

from src.data.extract_and_reframe_serie import main_data_processing


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True),
                default="/users/az02234/Documents/Personnal_Git/PredictiveMaintenanceAutoEncoder/data/external/")
@click.argument('output_filepath', type=click.Path(),
                default="/users/az02234/Documents/Personnal_Git/PredictiveMaintenanceAutoEncoder/data/interim/")
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    print("Creating dataset")
    print("Input filepath: ", input_filepath)
    print("Output filepath: ", output_filepath)

    # Data should contain a value and a time stamp
    data = pd.read_csv(output_filepath+"data_dl.csv", dtype={'dataValue': np.float64, 'pji': np.int64},
                       parse_dates=['sourceTimestamp_dtformat'], nrows=500000)
    data, max_length, num_serie = main_data_processing(data)
    print("\nMaximum length of a serie is:", max_length, "\nNumber of series is:", num_serie)
    print("Data transformed from csv to np.array")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
