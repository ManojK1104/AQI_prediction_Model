import numpy as np
import pandas as pd

def load():
    df=pd.read_excel(r'D:\AQI_prediction_Model\data\AirQualityUCI.xlsx')

    return df