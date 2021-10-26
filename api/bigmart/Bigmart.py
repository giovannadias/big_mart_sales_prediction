import pickle
import inflection
import pandas as pd
import numpy as np
import math


class Bigmart(object):
    def __init__(self):
        self.home_path = '/Users/Giovana/repos/big_mart_sales_prediction/'
        self.item_visibility_scaler      = pickle.load(open(self.home_path + 'parameter/item_visibility_scaler.pkl', 'rb'))
        self.item_mrp_scaler             = pickle.load(open(self.home_path + 'parameter/item_mrp_scaler.pkl', 'rb'))
        self.item_weight_scaler          = pickle.load(open(self.home_path + 'parameter/item_weight_scaler.pkl', 'rb'))
        self.outlet_year_scaler          = pickle.load(open(self.home_path + 'parameter/outlet_year_scaler.pkl', 'rb'))
        self.item_identifier_scaler      = pickle.load(open(self.home_path + 'parameter/item_identifier_scaler.pkl', 'rb'))
        self.outlet_location_type_scaler = pickle.load(open(self.home_path + 'parameter/outlet_location_type_scaler.pkl', 'rb'))
        self.outlet_type_scaler          = pickle.load(open(self.home_path + 'parameter/outlet_type_scaler.pkl', 'rb'))


    def data_cleaning(self, df1): 
        ## 1.1 Rename Columns 

        cols_old = ['Item_Identifier', 'Item_Weight', 'Item_Fat_Content', 'Item_Visibility',
                    'Item_Type', 'Item_MRP', 'Outlet_Identifier',
                    'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type',
                    'Outlet_Type']
        snakecase = lambda x: inflection.underscore(x)
        cols_new = list(map(snakecase, cols_old))
        df1.columns = cols_new

        ## 1.5 Fillout NA 

        #========item_weight==============
        #create aux item dataframe dataframe
        aux1 = df1[['item_identifier', 'item_weight']].groupby(['item_identifier', 'item_weight']).max()
        df_aux_item = pd.DataFrame(aux1).reset_index()

        #merge aux item dataframe with df1
        df1= pd.merge(df1, df_aux_item, how='left', on='item_identifier')
        df1.drop('item_weight_x', inplace=True, axis=1)
        df1 = df1.rename(columns={'item_weight_y': 'item_weight'})

        #substitute NA left with mean products weight
        mean_weight = df1['item_weight'].mean()
        df1['item_weight'] = df1['item_weight'].fillna(mean_weight)

        #============outlet_size=========== 
        #replace outlet_identifier == OUT010 size for small based on similar OUT019
        df1.loc[df1['outlet_identifier'] == 'OUT010', 'outlet_size'] = 'Small'

        #replace outlet_identifier == OUT017 size for small based on similar majority for Supermarket Type1
        df1.loc[df1['outlet_identifier'] == 'OUT017', 'outlet_size'] = 'Small'

        #replace outlet_identifier == OUT045 size for small based on similar majority for Supermarket Type1
        df1.loc[df1['outlet_identifier'] == 'OUT045', 'outlet_size'] = 'Small'

        ## 1.6 Change variables names
        
        #============item_fat_content=========== 
        #replace low fat to Low Fat
        df1.loc[df1['item_fat_content'] == 'low fat', 'item_fat_content'] = 'Low Fat'

        #replace LF to Low Fat
        df1.loc[df1['item_fat_content'] == 'LF', 'item_fat_content'] = 'Low Fat'

        #replace reg to Regular
        df1.loc[df1['item_fat_content'] == 'reg', 'item_fat_content'] = 'Regular'
        
        return df1
    
    def feature_engineering(self, df2):
        #outlet_id
        df2['outlet_id'] = df2['outlet_identifier'].str.extract(r'(\d{2}$)').astype(int)
        
        return df2
    
    def data_preparation(self, df4): 
        #rescaling
        #item_visibility
        df4['item_visibility'] = self.item_visibility_scaler.fit_transform(df4[['item_visibility']].values)

        #item_mrp
        df4['item_mrp'] = self.item_mrp_scaler.fit_transform(df4[['item_mrp']].values)

        #item_weight
        df4['item_weight'] = self.item_weight_scaler.fit_transform(df4[['item_weight']].values)

        #outlet_establishment_year
        df4['outlet_year'] = self.outlet_year_scaler.fit_transform(df4[['outlet_establishment_year']].values)

        ## 4.2 Transformation
        ###  4.2.1 Enconding
        #item_identifier - label enconding
        df4['item_identifier'] = self.item_identifier_scaler.fit_transform(df4['item_identifier'])

        #item_fat_content - one hot enconding
        df4 = pd.get_dummies(df4, prefix=['item_fat_content'], columns=['item_fat_content'])

        #item_type - label enconding
        type_dict = {'Dairy': 1,
                     'Soft Drinks': 2, 
                     'Meat': 3, 
                     'Fruits and Vegetables': 4,
                     'Household': 5, 
                     'Baking Goods': 6, 
                     'Snack Foods': 7, 
                     'Frozen Foods': 8,
                     'Breakfast': 9, 
                     'Health and Hygiene': 10, 
                     'Hard Drinks': 11, 
                     'Canned': 12,
                     'Breads': 13, 
                     'Starchy Foods': 14, 
                     'Others': 15, 
                     'Seafood': 16}
        df4['item_type'] = df4['item_type'].map(type_dict)

        #outlet_size - ordinal enconding
        size_dict = {'Small': 0, 'Medium': 1, 'High': 2}
        df4['outlet_size'] = df4['outlet_size'].map(size_dict)

        #outlet_location_type - label enconding
        df4['outlet_location_type'] = self.outlet_location_type_scaler.fit_transform(df4['outlet_location_type'])

        #outlet_type - label enconding 
        df4['outlet_type'] = self.outlet_type_scaler.fit_transform(df4['outlet_type'])
        
        cols_selected = ['item_mrp', 'outlet_type', 'outlet_size', 'outlet_id', 'item_identifier', 'outlet_location_type', 'item_visibility', 'outlet_year']

        
        return df4[cols_selected]
    
    def get_prediction(self, model, original_data, test_data):
        #prediction
        pred = model.predict(test_data)
        
        #join pred into the original data
        original_data['prediction'] = np.expm1(pred)
        
        return original_data.to_json(orient='records')
