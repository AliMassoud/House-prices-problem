import string
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import house_prices.process as preprocess_py
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import mean_squared_log_error


train_data_path_after_split = "../splitted_data/Train_after_split.csv"
test_data_path_after_split = "../splitted_data/Test_after_split.csv"
model_name = '../Models/model.sav'


def Save_(x: pd.DataFrame, y: pd.DataFrame, filepath: string):
    my_data = pd.concat([
        pd.DataFrame(x),
        pd.DataFrame(y)
    ], axis=1)
    my_data.to_csv(filepath)


def rmsle(y_test: np.ndarray, y_pred: np.ndarray, precision: int = 2) -> float:
    rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
    return round(rmsle, precision)


def build_model(data: pd.DataFrame) -> dict[str, str]:
    Y = data['SalePrice']
    X = data.drop(['SalePrice'], axis=1)

    # set the seed!
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,
                                                        random_state=0)
    Save_(X_train, y_train, train_data_path_after_split)

    Ord_train, Not_Ord_train, numeric_train = preprocess_py.split_(X_train)
    x_train = preprocess_py.preprocess(Ord_train, Not_Ord_train,
                                       numeric_train, dataset_Typee=False)

    Ord_test, Not_Ord_test, numeric_test = preprocess_py.split_(X_test)
    x_test = preprocess_py.preprocess(Ord_test, Not_Ord_test,
                                      numeric_test, dataset_Typee=True)

    mod = RandomForestRegressor()
    mod.fit(x_train, y_train)

    joblib.dump(mod, model_name)

    y_pred = mod.predict(x_test)
    result = rmsle(y_test, y_pred)

    return {"RMSE": result}
