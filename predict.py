#!/usr/bin/env python
# -*- coding: utf-8 -*-

# predict.py
import pandas as pd
import joblib

def main(args):
    # Load the test dataset
    test_data = pd.read_csv(args.data)

    # Load the trained model
    model = joblib.load(args.model)

    # Make predictions
    predictions = model.predict(test_data)

    # Save predictions to a CSV file
    submission = pd.DataFrame({'Id': test_data.index, 'Prediction': predictions})
    submission.to_csv(args.output, index=False)

if __name__ == '__main__':
    
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data/hidden_test.csv')
    parser.add_argument('--model', type=str, default='model.pkl')
    parser.add_argument('--output', type=str, default='predictions.csv')
    args = parser.parse_args()
    print(args.__dict__)
    
    main(args)
