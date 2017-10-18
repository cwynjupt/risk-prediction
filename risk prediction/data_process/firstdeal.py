#!usr/bin/env python3
# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt


def get_data():
    entbase = pd.read_csv('C:\\Users\\dell\\Desktop\\大数据竞赛\\data fountain\\企业预测\\data\\BDCI2017-liangzi-10.12\\1entbase.csv')
    train = pd.read_csv('C:\\Users\\dell\\Desktop\\大数据竞赛\\data fountain\\企业预测\\data\\BDCI2017-liangzi-10.12\\train.csv')
    evaluation = pd.read_csv('C:\\Users\\dell\\Desktop\\大数据竞赛\\data fountain\\企业预测\\data\\BDCI2017-liangzi-10.12\\evaluation_public.csv')

    remove_and_fill(entbase)

    train_eid = train['EID']
    train_target = train['TARGET']
    train_feature = entbase[entbase['EID'].isin(train_eid)].drop(['EID', 'RGYEAR'], axis=1)
    evaluation_eid = evaluation['EID']
    evaluation_feature = entbase[entbase['EID'].isin(evaluation_eid)].drop(['EID', 'RGYEAR'], axis=1)
    return train_eid, train_feature, train_target, evaluation_eid, evaluation_feature


def remove_and_fill(df):
    numerical_col = [col for col in df.columns if df[col].dtype != 'object']
    categorical_col = [col for col in df.columns if df[col].dtype == 'object']
    df[numerical_col] = df[numerical_col].fillna(0)
    for col in categorical_col:
        df[col] = df[col].astype('category')
        if df[col].isnull.any():
            df[col] = df[col].cat.add_categories(['MISSING'])
            df[col] = df[col].fillna('MISSING')
    return df


