from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import dataset
import time
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style='darkgrid')


def isolation():

    # Create dictionaries and lists to store models and generated data
    Dataframes_xtrain = {}
    Dataframes_xtest = {}
    y_test_true = {}
    models = {}
    anomaly = {}
    auc_scores = {}
    training_times = {}
    prediction_times = {}
    auc_list = [] # list to store data for plotting
    std_auc = []
    mean_auc = []

    first_random_state = 42
    
    #load and preprocess the data
    x,y,label_map = dataset.data_preprocess(filename='KDDTrain.csv')

    if x is None:
        print("Preprocessing failed")
        return 
    
    #labeling outcome
    normal_label = 'attack_class'
    # Use np.where to convert labels to binary (0 or 1)
    y = np.where(y == normal_label, 0, 1)

    # Fractions for model training 
    test_size_fractions = np.arange(0.15,0.61,0.05)
    
    #dataset spliting
    for i in range(10):
        Dataframes_xtrain[i] = {}
        Dataframes_xtest[i] = {}
        y_test_true[i] = {}
        for j in range(10):
            test_split = test_size_fractions[j]
            #stratify is crucial for balanced classes
            xtrain, xtest, ytrain_split, ytest_split = train_test_split(x,y,test_size=test_split,random_state=first_random_state+i,stratify=y)

            Dataframes_xtrain[i][j] = xtrain
            Dataframes_xtest[i][j] = xtest
            y_test_true[i][j] = ytest_split
        
        

    #Model training loop
    for i in range(10):
        models[i] = {}
        training_times[i] = {}
        for j in range(10):
            data_usage = test_size_fractions[j]
            # working with contamination parameter
            model = IsolationForest(n_estimators=100,contamination=0.20,random_state=first_random_state) 
            try:
                start_time = time.time()
                model.fit(Dataframes_xtrain[i][j])
                training_time = time.time() - start_time
                print(f"Model {i+1} has been trained with {1-data_usage:.2f} % usage of dataset")
            except Exception as e:
                print(f"Error during training model {i+1}: {e}")
                training_time = 0
            
            models[i][j] = model
            training_times[i][j] = training_time

    #Model prediction and evaluation (ROC AUC) loop
    for i in range(10):
        prediction_times[i] = {}
        auc_scores[i] = {}
        for j in range(10):
            try:
                start_time = time.time()
                #using -scores as IsolationForest assigns lower scores to anomalies
                scores = -models[i][j].decision_function(Dataframes_xtest[i][j])
                prediction_time = time.time()-start_time
            except Exception as e:
                print(f"Error during prediction model {i+1}: {e}")
                scores = np.zeros_like(y_test_true[i][j])
                prediction_time = 0
            
            prediction_times[i][j] = prediction_time
            y_true = y_test_true[i][j]
            
            if len(np.unique(y_true)) > 1: # check if at least two classes exist, i.e., 0 and 1
                auc = roc_auc_score(y_true,scores) # if they exist, calculate the roc_auc score
            else:
                auc = 0.5
                #print(f"Warning: Model {i+1} (Usage: {1-j:.2f} %). Only one labels in test. AUC set to 0.5")

            auc_scores[i][j] = auc
            # data printing (commented out in original)
            #print(f"ROC AUC for model {i+1} (Train Usage: {1-j:.2f} %) is: {auc:.4f}")

            auc_list.append({
                'Model': i+1,
                'Training Ratio':1-test_size_fractions[j],
                'ROC AUC Score': auc
            })

    #Summary statistics (best score and shortest time)
    best_auc = 0
    best_auc_model = None
    best_auc_ratio = None

    for i in auc_scores:
        for j in auc_scores:
            if auc_scores[i][j] > best_auc:
                best_auc = auc_scores[i][j]
                best_auc_model = i+1
                best_auc_ratio = 1-test_size_fractions[j]
    print(f'\nBest ROC AUC Score: {best_auc:.4f} , (Model {best_auc_model}, Training set Ratio: {best_auc_ratio:.2f})')

    shortest_training_time = min([min(t.values()) for t in training_times.values()])
    print(f"The shortest training time was: {shortest_training_time:.4f} seconds")

    #Plotting results (learning curve and score distribution)
    df_results = pd.DataFrame(auc_list) # create a dataframe

    #calculate mean and std dev for each training ratio
    for j in range(10):
        mean_auc.append(np.mean([auc_scores[i][j] for i in range(10)]))
        std_auc.append(np.std([auc_scores[i][j] for i in range(10)]))

    fig, axes = plt.subplots(1,2, figsize=(18,7))
    train_ratio_plot = 1-test_size_fractions

    #plot A: learning curve(roc auc)
    axes[0].plot(train_ratio_plot,mean_auc,marker='o',color='red',label='Mean ROC AUC')
    #fill between (mean +- Std Dev)
    axes[0].fill_between(train_ratio_plot,np.array(mean_auc) - np.array(std_auc),
                         np.array(mean_auc) + np.array(std_auc),
                         color='red',alpha=0.1,label='Std Dev')
            
    sns.scatterplot(ax=axes[0],data=df_results,x='Training Ratio',y='ROC AUC Score',
        hue='Model',palette='viridis',legend=False,alpha=0.6)
            
    # Reference line AUC=0.5
    axes[0].axhline(0.5,color='black',linestyle='--',linewidth=1,label='Random Guess(AUC=0.5)')

    axes[0].set_title("A) Learning Curve: ROC AUC Score vs Training Ratio",fontsize=14)
    axes[0].set_xlabel('Training Ratio (%)', fontsize=12)
    axes[0].set_ylabel('ROC AUC Score', fontsize=12)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(loc='lower right')

    # Selecting a typical iteration (e.g., i=0, medium test size j=0.30)
    i_sample = 0
    j_sample = 3 # Test Size 30% -> Train Usage 70%

    # Get the scores. We use -scores so anomalies have a higher value.
    sample_scores = -models[i_sample][j_sample].decision_function(Dataframes_xtest[i_sample][j_sample])
    sample_y_true = y_test_true[i_sample][j_sample]
    
    df_scores = pd.DataFrame({
        'Anomaly_Score': sample_scores,
        'True_Label': np.where(sample_y_true == 1, 'Attack (1)', 'Normal (0)')
    })

    # Use histplot to show the distribution of scores per class
    sns.histplot(ax=axes[1], data=df_scores, x='Anomaly_Score', hue='True_Label', 
                 palette={'Normal (0)': 'skyblue', 'Attack (1)': 'red'}, 
                 kde=True, bins=50)

    axes[1].set_title(f'B) Anomaly Scores Distribution (Overlap)', fontsize=14)
    axes[1].set_xlabel('Anomaly Score (Higher Score = Anomaly)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    
    # Finally, display and save the figure
    plt.tight_layout() # Adjusts subplots to prevent overlapping
    plt.savefig('IsolationForest_Evaluation.png') # Save as a single image
    plt.show()

    return auc_scores

if __name__ == "__main__":
    isolation()