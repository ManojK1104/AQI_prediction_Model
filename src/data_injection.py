import numpy as np
import pandas as pd

def load():
    df=pd.read_excel(r'https://raw.githubusercontent.com/ManojK1104/AQI_prediction_Model/main/data/AirQualityUCI.xlsx')

    return df