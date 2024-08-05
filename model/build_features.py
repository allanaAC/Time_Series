import pandas as pd
import logging

# 
def create_features(df):
    try:
       
        # Converting the target variable into a categorical variable
        df['Admit_Chance']=(df['Admit_Chance'] >=0.8).astype(int)
        # Dropping columns
        df = df.drop(['Serial_No'], axis=1)
        df.describe().T
        # Create dummy variables for all 'object' type variables except 'Loan_Status'
        df = pd.get_dummies(drop_first=, columns=['University_Rating','Research'])
        return x,y
    
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))