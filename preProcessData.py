# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:02:48 2022

@author: ted
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder ,StandardScaler,LabelEncoder


def preProcessData(df) :
    
    # Correct NaN value CryoSleep
    df['CryoSleep'] = np.where( (df['CryoSleep'].isnull()) & 
                               ((df['RoomService'] == 0.0) & (df['FoodCourt'] == 0.0) & 
                                (df['ShoppingMall'] == 0.0) & (df['Spa'] == 0.0) & 
                                (df['VRDeck'] == 0.0)), True, df['CryoSleep'])
# =============================================================================
#     df['CryoSleep'] = np.where( (df['CryoSleep'].isnull()) & 
#                                (((df['RoomService'] == 0.0) | df['RoomService'].isnull()) & 
#                                 ((df['FoodCourt'] == 0.0)) | df['FoodCourt'].isnull() & 
#                                 ((df['ShoppingMall'] == 0.0) | df['ShoppingMall'].isnull()) & 
#                                 ((df['Spa'] == 0.0) | df['Spa'].isnull()) & 
#                                 ((df['VRDeck'] == 0.0) | df['VRDeck'].isnull())), True, df['CryoSleep'])
# =============================================================================
    df['CryoSleep'] = np.where((df['CryoSleep'].isnull()) & 
                               ((df['RoomService'] > 0.0) | (df['FoodCourt'] > 0.0) | 
                                (df['ShoppingMall'] > 0.0) | (df['Spa'] > 0.0) | 
                                (df['VRDeck'] > 0.0)), False, df['CryoSleep'])

    # Correct NaN value for the amenities
    amenity_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    
    
    for service in amenity_cols:

        df[service] = np.where((df[service].isnull()) & (df['CryoSleep'] == True), 0.0, df[service])
        
        df[service] = df[service].fillna(0)
        
        # NaN填入出現次數最多的值
        #df[service] = df[service].fillna(df[service].mode().iloc[0])
 
    # split_cabins_data
    df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mode().iloc[0])
    a = df['Cabin'].str.split('/', expand=True)
    df[['Deck','Num', 'Side']] = df['Cabin'].str.split('/', expand=True)
    b =    df['PassengerId'].str.split('_', expand=True)
    df[['PGroup','PNr']] = df['PassengerId'].str.split('_', expand=True)
    
    df.drop(['Cabin', 'PassengerId','Name','VIP'], axis=1, inplace=True)
    
    # one hot encoding
    df = pd.get_dummies(df, columns=['Destination'], prefix="D")
    df = pd.get_dummies(df, columns=['HomePlanet'], prefix="H")
    df = pd.get_dummies(df, columns=['Side'])
    df = pd.get_dummies(df, columns=['CryoSleep'])
    df = pd.get_dummies(df, columns=['Deck'])
    
    # convert data type
    df['Num'] = df['Num'].astype('int')
    df['PNr'] = df['PNr'].astype('int')
    df['PGroup'] = df['PGroup'].astype('int')
    #df['CryoSleep'] = df['CryoSleep'].astype('int')
    #df['VIP'] = df['VIP'].fillna(df['VIP'].mode().iloc[0])
    #df['VIP'] = df['VIP'].astype('int')
    
    # drop NaN row
    df.fillna(value=df['Age'].mean(), inplace=True)
    df.fillna(value=df['Num'].median(), inplace=True)
    
    # add Total Consumption feature
    consume_feats = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    consume_more_for_alive = ['FoodCourt', 'ShoppingMall']
    consume_more_for_die = ['RoomService', 'Spa', 'VRDeck']
    
    df['ConsumptionTotal'] = df.loc[:, consume_feats].sum(axis=1)
    #df['ConsumptionAlive'] = df.loc[:, consume_more_for_alive].sum(axis=1)
    #df['ConsumptionDie'] = df.loc[:, consume_more_for_die].sum(axis=1)

    return df 
    
def main():
    path = './data/'
    train_df = pd.read_csv( path + 'train.csv')
    test_df = pd.read_csv( path + 'test.csv')
    
    train_processed = preProcessData(train_df)
    test_processed = preProcessData(test_df)
    
    # get label to temp
    temp = pd.DataFrame()
    temp['Transported'] = train_processed['Transported'].astype('int')
    train_processed.drop(['Transported'], axis=1, inplace=True)
    
    # normalization without label
    scale = StandardScaler() #z-scaler物件
    train_processed = pd.DataFrame(scale.fit_transform(train_processed), columns=train_processed.keys())
    test_processed = pd.DataFrame(scale.fit_transform(test_processed), columns=test_processed.keys())
    
    # add label to the end of data
    train_processed['Transported'] = temp
    
    # save 
    train_processed.to_csv(path + 'train_processed.csv',index=False)
    test_processed.to_csv(path + 'test_processed.csv',index=False)
    
    
    
if __name__ == "__main__":
    main()
