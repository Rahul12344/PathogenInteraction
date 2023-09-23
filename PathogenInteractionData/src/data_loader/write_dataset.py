def write_data(df, predictions, num):  
    df['Prediction {}'.format(num)] = predictions
    
    return df  