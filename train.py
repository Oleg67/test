#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# train.py
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib


def main(args):
    
    # Load the dataset
    data = pd.read_csv(args.train_data)

    # Split features and target
    X = data.drop(columns=[args.target])
    y = data[args.target]

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Validate the model
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f'Validation RMSE: {rmse}')

    # Save the model
    joblib.dump(model, args.output)
    print(f"Save trained model to {args.output}")
    
if __name__ == '__main__':
    
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='data/train.csv')
    parser.add_argument('--target', type=str, default='target')
    parser.add_argument('--output', type=str, default='model.pkl')
    args = parser.parse_args()
    print(args.__dict__)
    
    main(args)
