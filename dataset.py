import pandas as pd
import os 
import functions 
from sklearn.preprocessing import StandardScaler

def data_preprocess(filename):
    #Loads NSL-KDD dataset and apply necessary preprocessin steps

    #Define categorical and target columns for NSL-KDD
    #we assume these are the column names in kdd_test.csv based on standard NSL-KDD structure
    target_column = 'attack_class'
    categorical_columns = ['protocol_type','service','flag'] #standard NSL-KDD categorical features
    #add any columns we would like to drop(e.g 'ID')
    columns_to_drop = ['num_outbound_cmds']

    #load the file
    try:
        df = pd.read_csv(filename,delimiter=';')
        print(f"File {filename} has been loaded")
    except FileNotFoundError as e:
        print(e)
        return None
    
    #Handle missin values(Imputation)
    df = functions.fill_missing_values(df)

    #drop unused columns(if any are defined)
    if columns_to_drop:
        df = functions.drop_train_unused(df,columns_to_drop)
    
    #seperate features (X) and target (y)
    y_df = df[[target_column]].copy()
    X_df = df.drop(columns=[target_column],errors='ignore')

    #encode y - required for classification
    y_df, label_map = functions.encode_label(y_df,target_column)
    y_encoded = y_df.values.ravel()

    #one-hot-encodig for categorical features 
    X_categorical = X_df[categorical_columns]
    X_numerical = X_df.drop(columns=categorical_columns, errors='ignore')

    #one-hot-encoding, drop_first=True avoid multicollinearity
    X_categorical_encoded = pd.get_dummies(X_categorical,columns=categorical_columns,drop_first=True,dtype=int)
    X_intermediate = pd.concat([X_numerical,X_categorical_encoded],axis=1)
    print("Categorical columns have been One-Hot Encoded")

    #Scaling numerical features
    #apply standad scaling to all features(which are now all numerical)
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_intermediate)

    #convert back to dataframe
    X_final = pd.DataFrame(X_scaled_array,columns=X_intermediate.columns)
    print("All features have been Standard Scaled")

    print("Preprocessing completd successfully")

    return X_final,y_encoded,label_map

if __name__ == "__main__":
    x, y, mapping = data_preprocess(filename='kdd_test.csv')