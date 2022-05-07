import string
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from pickle import dump, load


ordinal_encoder_file = "../Models/Ordinal_Encoder.pkl"
scaler_file = "../Models/Scaler.pkl"
one_hot_encoder_file = "../Models/One_Hot_Encoder.pkl"


def get_Encoded_OneHot_Encoder(Not_Ord: pd.DataFrame) -> pd.DataFrame:
    cat_cols = Not_Ord.columns.values
    cols_encoded = []
    for col in cat_cols:
        cols_encoded += [f"{col}_{cat}" for cat in list(Not_Ord[col].unique())]

    oh_encoder = load(open(one_hot_encoder_file, "rb"))
    encoded_cols = oh_encoder.transform(Not_Ord[cat_cols])
    # oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # oh_encoder = load(open(one_hot_encoder_file, "rb"))
    # encoded_cols = oh_encoder.fit_transform(Not_Ord[cat_cols])
    # print(encoded_cols[:10])
    df_enc = pd.DataFrame(encoded_cols,
                          columns=oh_encoder.get_feature_names_out())
    dump(oh_encoder, open(one_hot_encoder_file, "wb"))
    return df_enc


def Encoded_Ordinal(categorical_features_Ord: pd.DataFrame) -> pd.DataFrame:
    enc = load(open(ordinal_encoder_file, "rb"))
    temp = enc.transform(categorical_features_Ord)

    cat_encoded = pd.DataFrame()
    finall = pd.concat([
        cat_encoded,
        pd.DataFrame(
            temp,
            columns=[categorical_features_Ord.columns.values]
        )
        ], axis=1)
    dump(enc, open(ordinal_encoder_file, "wb"))
    return finall


# 1. Function to replace NAN values with mode value
def impute_most_freq(DataFrame: pd.DataFrame, ColName: string):
    # .mode()[0] - gives first category name
    median = DataFrame[ColName].mode()[0]
    most_frequent_category = median
    # replace nan values with most occured category
    DataFrame[ColName] = DataFrame[ColName]
    DataFrame[ColName].fillna(most_frequent_category, inplace=True)


def split_(x: pd.DataFrame):
    perc = 50.0
    min_count = int(((100-perc)/100)*x.shape[0] + 1)
    x = x.drop(['Id', 'Utilities', 'FireplaceQu', 'MasVnrType'], axis=1)
    X_final = x.dropna(axis=1, thresh=min_count)

    categorical_features = X_final.select_dtypes('object')

    Ordinal = categorical_features[['MSZoning', 'LandSlope', 'BldgType',
                                    'RoofMatl', 'ExterQual', 'ExterCond',
                                    'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                    'BsmtFinType1', 'BsmtFinType2',
                                    'HeatingQC', 'CentralAir',
                                    'Electrical', 'KitchenQual',
                                    'Functional', 'GarageFinish', 'GarageQual',
                                    'GarageCond', 'PavedDrive', 'SaleCondition'
                                    ]]

    Not_Ordinal = categorical_features[['Street', 'LotShape', 'LandContour',
                                        'LotConfig', 'Neighborhood',
                                        'Condition1', 'Condition2',
                                        'HouseStyle', 'RoofStyle',
                                        'Exterior1st', 'Exterior2nd',
                                        'Foundation', 'Heating',
                                        'GarageType', 'SaleType']]

    numeric = X_final.select_dtypes('number')

    return Ordinal, Not_Ordinal, numeric


def scale_final_data(x: pd.DataFrame, dataset_Type: bool) -> pd.DataFrame:
    if(dataset_Type is False):
        scaler = StandardScaler()
        scaler.fit(x)
        dump(scaler, open(scaler_file, "wb"))
    elif(dataset_Type is True):
        scaler = load(open(scaler_file, "rb"))
        scaler.transform(x)
    return x


def preprocess(Ordinal: pd.DataFrame, Not_Ordinal, numeric, dataset_Typee):
    for Columns in Ordinal.columns:
        impute_most_freq(Ordinal, Columns)

    for Columns in Not_Ordinal.columns:
        impute_most_freq(Not_Ordinal, Columns)

    if(dataset_Typee is False):
        enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                             unknown_value=np.nan)
        oh_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

        enc.fit(Ordinal)
        oh_encoder.fit(Not_Ordinal)

        dump(enc, open(ordinal_encoder_file, "wb"))
        dump(oh_encoder, open(one_hot_encoder_file, "wb"))

    ordinal = Encoded_Ordinal(Ordinal)
    Not_ordinal = get_Encoded_OneHot_Encoder(Not_Ordinal)

    ordinal.reset_index()
    numeric.reset_index()
    Not_ordinal.reset_index()
    numeric_norminal = numeric.join(ordinal)
    numeric_norminal = numeric_norminal.fillna(numeric_norminal.median())

    final = numeric_norminal[['OverallQual', 'YearBuilt', 'YearRemodAdd',
                             'MasVnrArea', 'TotalBsmtSF', '1stFlrSF',
                              'GrLivArea', 'FullBath', 'TotRmsAbvGrd',
                              'GarageYrBlt', 'GarageCars']].join(Not_ordinal)
    final.interpolate(method='linear', limit_direction='forward', inplace=True)
    final.interpolate(method='linear', limit_direction='backward',
                      inplace=True)

    x = scale_final_data(final, dataset_Typee)
    return x
