import numpy as np
import pandas as pd 
from xgboost import XGBClassifier
import dataset
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score
from sklearn.preprocessing import RobustScaler,LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def xgbclassifier():

    #αρχικοποιηση των dictionaries, και λιστων που θα αποθηκευονται δεδομενα
    Dataframe_xtrain = {}
    Dataframe_xtest = {}
    Dataframe_ytrain = {}
    Dataframe_ytest = {}
    output_predictions = {}
    models = {}
    accuracies = {}
    precisions = {}
    recalls = {}
    f1_scores = {}
    training_times = {}
    prediction_times = {}
    mean_acc = []
    mean_pre = []
    mean_rec = []
    mean_f1 = []
    std_acc = []
    std_pre = []
    std_rec = []
    std_f1 = []
    first_random_state = 42
    
    #αρχικα φορτωνουμε το αρχειο στο df
    x,y,label_map = dataset.data_preprocess(filename='KDDTrain.csv')

    if x is None:
        print(f"Preprocessing failed")
        return None
    
    # x is already a Dataframe
    x_df = x.reset_index(drop=True)
    #transform y_array(numpy) to pandas series
    y_series = pd.Series(y,name='Attack_class')
    #finding each label
    class_counts = y_series.value_counts()
    #finding labels appears only once
    rare_classes = class_counts[class_counts == 1].index
    
    if len(rare_classes) > 0:
        print(f"Found {len(rare_classes)} classes with only 1 sample. Removing them...")
        print(f"Classes to be removed: {list(rare_classes)}")
        #finding indices to lines not being rare class 
        indices_to_keep = y_series[~y_series.isin(rare_classes)].index

        x_new = x_df.loc[indices_to_keep].values
        y_new = y_series.loc[indices_to_keep].values

        x = x_new
        y = y_new

        print(f"New shape after removing rare classes: {x.shape}")
    else:
        print("No rare (1-sample) classes found. Proceeding.")

    y = y-1

    #array specifying the percentage of the dataset used for trainig the model(based on test_size)    
    train_size = np.arange(0.15,0.61,0.05)

    #διαχωρισμος του dataset
    for i in range(10):
        Dataframe_xtrain[i] = {}
        Dataframe_xtest[i] = {}
        Dataframe_ytrain[i] = {}
        Dataframe_ytest[i] = {}
        for j in range(10):
            test_split = train_size[j]
            x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_split,random_state=first_random_state+i,stratify=y)
            
            #αποθηκευω τα δεδομενα 
            Dataframe_xtrain[i][j] = x_train
            Dataframe_xtest[i][j] = x_test
            Dataframe_ytrain[i][j] = y_train
            Dataframe_ytest[i][j] = y_test    
    
    #εκπαιδευση του μοντελου 
    for i in range(10):
        models[i] = {}
        training_times[i] = {}
        for j in range(10):
            test_split = train_size[j]
            #φορτωση του αλγοριθμου
            model = XGBClassifier(objective='multi:softmax',learning_rate=0.3,n_estimators=500,#num_class=(len(np.unique(y))),
                        eval_metric='mlogloss',random_state=first_random_state)
            try:
                start_time = time.time()
                model.fit(Dataframe_xtrain[i][j],Dataframe_ytrain[i][j])
                training_time = time.time() - start_time
                print(f"Model {i+1} was trained with {1-test_split:.2f} % dataset usage")
                models[i][j] = model
                training_times[i][j] = training_time
            except RuntimeError:
                print(f"Error while training model {i+1}")
                return None

    #output predictions 
    for i in range(10):
        output_predictions[i] = {}
        prediction_times[i] = {}
        for j in range(10):
            try:
                start_time = time.time()
                output_predictions[i][j] = models[i][j].predict(Dataframe_xtest[i][j])
                predict_time = time.time()-start_time
            except RuntimeError as e:
                print(e)
            prediction_times[i][j] = predict_time

    #Calculation of model metrics
    for i in range(10):
        accuracies[i] = {}
        precisions[i] = {}
        recalls[i] = {}
        f1_scores[i] = {}
        for j in range(10):
            test_split = train_size[j]
            accuracies[i][j] = accuracy_score(output_predictions[i][j],Dataframe_ytest[i][j])
            precisions[i][j] = precision_score(output_predictions[i][j],Dataframe_ytest[i][j],average='weighted',zero_division=1)
            recalls[i][j] = recall_score(output_predictions[i][j],Dataframe_ytest[i][j],average='weighted',zero_division=1)
            f1_scores[i][j] = f1_score(output_predictions[i][j],Dataframe_ytest[i][j],average='weighted',zero_division=1)

    # Finding best accuracy
    best_acc = 0
    best_acc_model = None
    best_acc_ratio = None
    for i in accuracies:
        for j in accuracies:
            test_split = train_size[j]
            if accuracies[i][j] > best_acc:
                best_acc = accuracies[i][j]
                best_acc_model = i+1
                best_acc_ratio = 1 - test_split
    print(f"Best Accuracy {best_acc:.4f} (Model {best_acc_model}, Training set Ratio {best_acc_ratio:.2f})")

    # Finding best precision
    best_pre = 0
    best_pre_model = None
    best_pre_ratio = None
    for i in precisions:
        for j in precisions:
            test_split = train_size[j]
            if precisions[i][j] > best_pre:
                best_pre = precisions[i][j]
                best_pre_model = i+1
                best_pre_ratio = 1 - test_split
    print(f"Best Precision {best_pre:.4f}, (Model {best_pre_model}, Training set Ratio {best_pre_ratio:.2f})")

    # Finding best recall 
    best_rec = 0
    best_rec_model = None
    best_rec_ratio = None
    for i in recalls:
        for j in recalls:
            test_split = train_size[j]
            if recalls[i][j] > best_rec:
                best_rec = recalls[i][j]
                best_rec_model = i+1
                best_rec_ratio = 1 - test_split
    print(f"Best Recall {best_rec:.4f} (Model {best_rec_model}, Training set Ratio {best_rec_ratio:.2f})")

    # Finding best F1
    best_f1 = 0
    best_f1_model = None
    best_f1_ratio = None
    for i in f1_scores:
        for j in f1_scores:
            test_split = train_size[j]
            if f1_scores[i][j] > best_f1:
                best_f1 = f1_scores[i][j]
                best_f1_model = i+1
                best_f1_ratio = 1 - test_split
    print(f"Best F1-Score {best_f1:.4f} (Model {best_f1_model}, Training set Ratio {best_f1_ratio:.2f})")

    #calculatin the model that required the shortest training time
    min_time = 100
    min_time_model = None
    min_time_ratio = None
    for i in training_times:
        for j in training_times:
            test_split = train_size[j]
            if training_times[i][j] < min_time:
                min_time = training_times[i][j]
                min_time_model = i+1
                min_time_ratio = 1 - test_split
    print(f"Shortest time {min_time} (Model {min_time_model}, Training set Ratio {min_time_ratio:.2f})")


    
    #finding the mean values for each metric to create arrays for plotting
    for j in range(10):
        #mean accuracy for each model
        avg_acc = np.mean([accuracies[i][j] for i in range(10)])
        mean_acc.append(avg_acc)
        std_acc.append(np.std([accuracies[i][j] for i in range(10)]))
        #mean precision for each model
        avg_pre = np.mean([precisions[i][j] for i in range(10)])
        mean_pre.append(avg_pre)
        std_pre.append(np.std([precisions[i][j] for i in range(10)]))
        #mean recall for each model
        avg_rec = np.mean([recalls[i][j] for i in range(10)])
        mean_rec.append(avg_rec)
        std_rec.append(np.std([recalls[i][j] for i in range(10)]))
        #mean f1 for each model
        avg_f1 = np.mean([f1_scores[i][j] for i in range(10)])
        mean_f1.append(avg_f1)
        std_f1.append(np.std([f1_scores[i][j] for i in range(10)]))

    
    # plotting the various datas
    fig, axs = plt.subplots(2, 2, figsize=(14,10))
    train_ratios = 1 - train_size
    #Accuracy
    axs[0,0].plot(train_ratios,mean_acc,marker='o',color='blue',label='Mean Accuracy')
    axs[0,0].fill_between(train_ratios,np.array(mean_acc) - np.array(std_acc),
                          np.array(mean_acc) + np.array(std_acc),color='blue',alpha=0.1,label='Std Dev')
    axs[0,0].set_title("Learning Curve (Accuracy)")
    axs[0,0].set_xlabel("Fraction of Training Data")
    axs[0,0].set_ylabel("Accuracy")
    axs[0,0].grid(True)
    axs[0,0].legend()

    #Recall
    axs[1,0].plot(train_ratios,mean_rec,marker='o',color='orange',label='Mean Recall')
    axs[1,0].fill_between(train_ratios,np.array(mean_rec) - np.array(std_rec),
                          np.array(mean_rec) + np.array(std_rec),color='orange',alpha=0.1,label='Std Dev')
    axs[1,0].set_title('Recall Score Curve')
    axs[1,0].set_xlabel("Training Set Ratio")
    axs[1,0].set_ylabel("Recall")
    axs[1,0].grid(True)
    axs[1,0].legend()

    #F1-Score
    axs[1,1].plot(train_ratios,mean_f1,marker='o',color='red',label='Mean F1-Score')
    axs[1,1].fill_between(train_ratios,np.array(mean_f1) - np.array(std_f1),
                          np.array(mean_f1) + np.array(std_f1),color='red',alpha=0.1,label='Std Dev')
    axs[1,1].set_title("F1 Score Curve")
    axs[1,1].set_xlabel("Training Score Ratio")
    axs[1,1].set_ylabel("F1 Score")
    axs[1,1].grid(True)
    axs[1,1].legend()

    #Precision
    axs[0,1].plot(train_ratios,mean_pre,marker='o',color='green',label='Mean Precision')
    axs[0,1].fill_between(train_ratios,np.array(mean_pre) - np.array(std_pre),
                          np.array(mean_pre) + np.array(std_pre),color='green',alpha=0.1,label='Std Dev')
    axs[0,1].set_title("Precision Score Curve")
    axs[0,1].set_xlabel("Training Set Ratio")
    axs[0,1].set_ylabel("Precision")
    axs[0,1].grid(True)
    axs[0,1].legend()

    plt.tight_layout()
    plt.savefig("XGBClassifier_Metrics.png")
    plt.show()

if __name__ == '__main__':
    xgbclassifier()
    

