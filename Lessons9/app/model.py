# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from pandas import Series
from sklearn.model_selection import train_test_split
import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from app import app


class TeachModels:

    __file_name = ''
    __model_name = ''

    def __init__(self, filename):
        self.__file_name = filename
        self.__model_name = f"model_{self.__file_name.split('.')[0]}"

    def series_factorizer(self, s: Series):
        series_res, unique = pd.factorize(s)
        reference = {x: i for x, i in enumerate(unique)}
        return series_res, reference

    def get_name_model(self):
        return self.__model_name

    def create(self):
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], self.__file_name), low_memory = False)

        df = df.loc[df.Dataset.isin([5, 6, 7, 8, 9])]
        df.drop('Dataset', axis=1, inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)

        NegClaimAmount = df.loc[df.ClaimAmount < 0, ['ClaimAmount', 'ClaimInd']]
        print('Unique values of ClaimInd:', NegClaimAmount.ClaimInd.unique())
        df.loc[df.ClaimAmount < 0, 'ClaimAmount'] = 0

        df['Gender'], GenderRef = self.series_factorizer(df['Gender'])

        df['MariStat'], MariStatRef = self.series_factorizer(df['MariStat'])

        VU_dummies = pd.get_dummies(df['VehUsage'], prefix='VehUsg', drop_first=False)

        df['SocioCateg'] = df['SocioCateg'].str.slice(0, 4)
        pd.DataFrame(df.SocioCateg.value_counts().sort_values()).rename({'SocioCateg': 'Frequency'}, axis=1)

        df = pd.get_dummies(df, columns=['VehUsage', 'SocioCateg'])
        df = df.select_dtypes(exclude=['object'])

        df['DrivAgeSq'] = df.DrivAge.apply(lambda x: x ** 2)
        df['LicAgeSq'] = df.LicAge.apply(lambda x: x ** 2)
        df['BonusMalusSq'] = df.BonusMalus.apply(lambda x: x ** 2)
        df['DrivAgeLog'] = df.DrivAge.apply(lambda x: np.log(x))
        df['LicAgeLog'] = df.LicAge.apply(lambda x: np.log(x))
        df['BonusMalusLog'] = df.BonusMalus.apply(lambda x: np.log(x))

        df.drop(["ClaimNbResp", "ClaimNbNonResp", "ClaimNbParking", "ClaimNbFireTheft", "ClaimNbWindscreen"], axis=1,
                inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(df.drop(['ClaimInd', 'ClaimAmount'], axis=1), df['ClaimInd'],
                                                            test_size=0.3, random_state=1)
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1)

        h2o.init()

        h2o_train = h2o.H2OFrame(pd.concat([x_train, y_train], axis=1))
        h2o_valid = h2o.H2OFrame(pd.concat([x_valid, y_valid], axis=1))
        h2o_test = h2o.H2OFrame(pd.concat([x_test, y_test], axis=1))

        h2o_train['ClaimInd'] = h2o_train['ClaimInd'].asfactor()
        h2o_valid['ClaimInd'] = h2o_valid['ClaimInd'].asfactor()
        h2o_test['ClaimInd'] = h2o_test['ClaimInd'].asfactor()

        glm = H2OGeneralizedLinearEstimator(family="binomial", link="logit", nfolds=5)
        glm.train(y="ClaimInd", x=h2o_train.names[:-1], training_frame=h2o_train, validation_frame=h2o_valid)

        train_pred = glm.predict(h2o_train).as_data_frame()
        valid_pred = glm.predict(h2o_valid).as_data_frame()
        test_pred = glm.predict(h2o_test).as_data_frame()

        h2o.save_model(glm, path=os.path.join(app.config['PATH_MODEL'], self.__model_name))
        h2o.shutdown()
