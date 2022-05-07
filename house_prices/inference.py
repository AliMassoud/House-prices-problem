import numpy as np
import pandas as pd
import house_prices.process as preprocess_py
import joblib


model_name = '../Models/model.sav'


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    X = input_data

    Ordinal, Not_Ordinal, numeric = preprocess_py.split_(X)
    x_test = preprocess_py.preprocess(Ordinal, Not_Ordinal, numeric,
                                      dataset_Typee=True)

    loaded_model = joblib.load(model_name)
    y_pred = loaded_model.predict(x_test)
    return y_pred
