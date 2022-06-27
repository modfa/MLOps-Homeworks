#!/usr/bin/env python
# coding: utf-8
# get_ipython().system('pip freeze | grep scikit-learn')

import pickle
import pandas as pd
import os
import sys


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']
# output_file = f'{year:04d}-{month:02d}.parquet'

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df



# feb_data_uri = 'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-02.parquet'
# # df = read_data('https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_????-??.parquet')
# df = read_data(feb_data_uri)



# year = 2021
# month = 2
# df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# df_result = pd.DataFrame()
# df_result['ride_id'] = df['ride_id']
# df_result['prediction'] = y_pred


# df_result.to_parquet(
#     output_file,
#     engine='pyarrow',
#     compression=None,
#     index=False
# )

def ride_duration_pred(year, month):
    data_uri = (f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')
    print(f'Reading the data from URI: {data_uri}')
    df = read_data(data_uri)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    print(f'Predicting the duration....Hold-On....')
    y_pred = lr.predict(X_val)
    return y_pred




def run():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    predictions = ride_duration_pred(year, month)
    print(f'The mean predicted duration for the given month and year is: {predictions.mean()}')


if __name__ == '__main__':
    run()



