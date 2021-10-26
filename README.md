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

After that, I made some quick analysis to get some vision about the data behavior in numerical and categorical attributes.

![item_outlet_displot](https://user-images.githubusercontent.com/82069205/138783940-8753e021-82ac-4fc3-bf62-938ea2820099.png)
![item_mrp_displot](https://user-images.githubusercontent.com/82069205/138783953-a3d69086-a7aa-4a9c-945b-8ab948867565.png)
![item_visibility_displot](https://user-images.githubusercontent.com/82069205/138783961-e7359ab7-4118-423e-bf57-be3c5c5fd5c9.png)

![categorical_boxplot](https://user-images.githubusercontent.com/82069205/138784076-f446fda7-fe65-4d99-95c5-fd1d3e7c94f2.PNG)

![outlets_boxplot](https://user-images.githubusercontent.com/82069205/138784085-240f90f4-34d6-4ed4-9731-878e0b53151f.PNG)

## ⚙ Feature Engineering
After this first analysis, I started to think about some hypothesis that could be proved by the data I had available, so I desined a mental map to help me find out some key themes that I could work on in order to get some hypothesis to be tested after. Finally, after creating the hypothesis, I would be able to create any needed new features so that I could start the exploratory data analysis.
![mindmap](https://user-images.githubusercontent.com/82069205/138784516-ec530c5e-5837-49a7-a5e5-ef4ef49021e8.png)

## 📊 Exploratory Data Analysis (EDA)
Using the mind map as a base, I created 10 hypothesis to be validated in the exploratory analysis phase. Puting an effort on finding out more information about the data, this phase of the project was crucial to get some insights and start to think about what would be the best features to be used for the model training. 
Where there is an resume about the hypothesis and witch of them were validated. 
- **H1** Bigger outlets should sell more.
- **H2** Tier 3 location outlets shoul sell more.
- **H3** Supermarket Type1 is the outlet type with more sales.
- **H4** Older outlets should sell more.
- **H5** Regular fat items should sell more.
- **H6** Items with bigger visibility should sell more.
- **H7** Household items should be more expensive.
- **H8** Expensive products should sell less.
- **H9** Lighter items should cost less.
- **H10** Heavier items should sell less.

![hypothesis_resume](https://user-images.githubusercontent.com/82069205/138785824-c7acc3bd-34e4-4e2e-acdc-71a0049dde77.PNG)

## 📍 Data Preparation
After getting some insights from data, I adjusted it so that it could be used in the models training. This phase contains the rescaling - for numerical attributes, enconding - for categorical attributes and the transformation of the response variable for logarithm. 

## 🎲 Feature Selection
In this phase, I used the Boruta algorithm to find out with features would be better for the model training. First, I separate the data into training and test, to then run Boruta.
After running, Boruta chose only two variables as relevant for the model: 'item_mrp' and 'outlet_type'.
In order to get a better performance, I select 6 other features to be add in the models traning. The selection was based on the exploratory analysis, in with the features in the hypothsis that I judged more relevants joined the final selecion. They were: 'outlet_size', 'outlet_id', 'item_identifier', 'outlet_location_type', 'item_visibility', 'outlet_year'

## 🤖 Machine Learning Models
After selecting which features would be used in the models training, I tested 4 different models, in order to get different results and compare which one would perform better.
The models I used where:
- Linnear Regression Model
- Linnear Regression Regularized Model (Lasso)
- Random Forest Regressor
- XGBoost Regressor
To get a baseline comparision, I also created an Average Model. After running all models, I performed the cross-validation method in each one, to get more accurate error results and compare the models performance.
![cross_validation_comparision](https://user-images.githubusercontent.com/82069205/138789518-791a6361-dd89-4b92-bd9f-554f0c39af5a.PNG)

I decided to go with the RMSE error value to decide the best model to use, since it is a better metric for models performance evaluation. The Random Forest Model got the smallest RMSE error, and since it's a technically easy and fast model to run locally in my computer, it was my final choice. 

## 🔦 Hyperparameters Fine Tuning
Since I didn't have more data fonts to work in order to get more features that could increase the model performance's, I used the hyperparameters fine tuning to get the best ones to use in the final model. I ran the random search using some parameters sample and the final model got the following results after the cross-validation.

![random_forest_final](https://user-images.githubusercontent.com/82069205/138790202-54099605-12e6-4875-a894-f7ce9e4d12a7.PNG)

## ⁉ Error Interpretation
To get a better explanation for a possible the non-technical audience, like business team, I got some business performance error explanation to facilitate the models performance understanding. This image contains a sample of 20 items with the prediction result, the worst and best scenario result and MAE and MAPE errors.
The worst and best scenarios where calculated by taking the prediction value, subtracting and adding the MAE value for each product, since the goal is to predict the sales for each product in each outlet.

![business_performance](https://user-images.githubusercontent.com/82069205/138791081-a4ab8b6b-ed46-4ff5-80c2-9893671eb740.png)

The MAE (Mean Absolute Error) tells about how much the predict data is wrong compared to the real value, in an absolute perspective. In order hand, the MAPE (Mean Absolute Percentage Error) tells about this same diffence, but from a percentage perspective. 

The final model got some big MAPE values - some of witch got an error rate of 700%. Since this was the first circle for the models implementation, this would be a major issue to be fixed in the next rounds of tests. Maybe getting more features would help to get a smaller error rate.

The final total business performance for this model prediction's was: 

![total_performance](https://user-images.githubusercontent.com/82069205/138792955-8d546041-4f47-4324-8c5c-8dc322bb813e.png)

The final model performance can be analysed by these 4 graphics, showing the differents between prediction and real values, error rate for each outlet, error density and predictions related with error - for residual analysis. 
![machine_learning_performance](https://user-images.githubusercontent.com/82069205/138793670-9ac31360-0714-4bdd-9176-1a33f73dc5da.PNG)

## 🛠 Deploy Model to Production
Finally, to get this model into production, I use Heroku to create a virtual environment, and prepared the whole model to fit into production.
The 3 steps to get the model ready for production was:
- Create "BigMart Class", with all the ETL phases of the project (data cleaning, feature engeneering, data preparation and prediction),
- Create the API Handler to set the data pipeline,
- API tester, to see if everything works fine.

## 📲 Telegram Bot 
In order to make better business decisions, the final user need to have an accessible tool to consult the model's predictions. For that, I created a telegram bot that gives the user the total sales prediction values for each outlet, only needing to get the outlet id that the user wants to see. Unfortunately, I wasn't able to set the bot to show each item sales prediction in each store, but that would be a next improvement for future bot's versions. You can acess the bot with this link https://t.me/bigmartsales_bot
