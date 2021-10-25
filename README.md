# Big Mart Sales Prediction 🛒

## 💡 Problem Statement
The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and predict the sales of each product at a particular outlet.

Using this model, BigMart will try to understand the properties of products and outlets which play a key role in increasing sales.

## 📚 Data Dictionary
We have train (8523) and test (5681) data set, train data set has both input and output variable(s). The goal was to predict the sales for test data set.

Variable	Description:
- Item_Identifier: Unique product ID
- Item_Weight: Weight of product
- Item_Fat_Content:	Whether the product is low fat or not
- Item_Visibility:	The % of total display area of all products in a store allocated to the particular product
- Item_Type:	The category to which the product belongs
- Item_MRP:	Maximum Retail Price (list price) of the product
- Outlet_Identifier:	Unique store ID
- Outlet_Establishment_Year:	The year in which store was established
- Outlet_Size:	The size of the store in terms of ground area covered
- Outlet_Location_Type:	The type of city in which the store is located
- Outlet_Type:	Whether the outlet is just a grocery store or some sort of supermarket
- Item_Outlet_Sales:	Sales of the product in the particular store. This is the outcome variable to be predicted.

## 🔍 Data Description
I started by making a data description in order to better understand the data I was going to work with. This phase is important so that we can check if there is any need to change the data columns names of if there is empty values in our dataframe and how we can figure that out.
In this case, there was some empty values in the 'item_weight' and 'outlet_size' columns. In each of them, I used a differente method to substitute the N/A values.
- Item_weight N/A values where replaced by the product's mean weight.
- Outlet_size N/A values where replaced by the value for outlets with similar caracteristics as 'Outlet_Location_Type' and 'Outlet_Type'. 
I also normalized the values in the 'item_fat_content' column so that it could only be rather 'Low Fat' or 'Regular', without no other type of writing.
After that, I made some quick analysis to get some vision about the data behavior.
![item_outlet_displot](https://user-images.githubusercontent.com/82069205/138780564-cfe100f9-d299-422a-ae6a-dfd705cc4359.png)



## ⚙ Feature Engineering


## 📊 Exploratory Data Analysis (EDA)

## 📍 Data Preparation

## 🎲 Feature Selection

## 🤖 Machine Learning Models

## 🔦 Hyperparameters Fine Tuning

## ⁉ Error Interpretation

## 🛠 Deploy Model to Production
