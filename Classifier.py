import pandas as pd
import time
from sklearn.model_selection import train_test_split # used to split the data into training and testing sets
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style='darkgrid')
import dataset

def classifier():

    # creating dictionaries to store dataframes
    Dataframes_xtrain = {}
    Dataframes_xtest = {}
    Dataframes_ytrain = {}
    Dataframes_ytest = {}
    output_prediction = {}
    accuracies = {}
    models = {}
    precisions = {}
    recalls = {}
    f1 = {}
    training_times = {}
    prediction_times = {}
    # dictionaries for plotting various data
    mean_acc = []
    mean_pre = []
    mean_rec = []
    mean_f1 = []
    std_acc = []
    std_pre = []
    std_rec = []
    std_f1 = []
    first_random_state = 42 # declared to set the initial seed for randomness

    #load and preprocess the data
    x,y,label_map = dataset.data_preprocess(filename='KDDTrain.csv')

    if x is None:
        print("Preprocessing failed.")
        return 

    # array specifying the percentage of the dataset used for training the model (based on test_size)
    train_size = np.arange(0.15,0.61,0.05)

    # in this segment, we perform the dataset splitting and store the various resulting datasets in variables
    for i in range(10):
        Dataframes_xtrain[i] = {}
        Dataframes_xtest[i] = {}
        Dataframes_ytrain[i] = {}
        Dataframes_ytest[i] = {}
        for j in range(10):    
            test_split = train_size[j]
            xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = test_split,random_state=first_random_state + i) # splitting the data
        
            # storing the data in a list/dictionary
            Dataframes_xtrain[i][j] = xtrain
            Dataframes_xtest[i][j] = xtest
            Dataframes_ytrain[i][j] = ytrain
            Dataframes_ytest[i][j] = ytest

    # in this segment, the model training is performed based on the datasets created above 
    for i in range(10):
        models[i] = {}
        training_times[i] = {}
        for j in range(10):
            # using the randomforestclassifier algorithm
            model = RandomForestClassifier(random_state=first_random_state)
            test_split = train_size[j]
            try:
                start_time = time.time()
                model.fit(Dataframes_xtrain[i][j],Dataframes_ytrain[i][j].values.ravel())
                training_time = time.time() - start_time
                print(f"Model {i+1} has been trained with training dataset {1-test_split:.2f} %")
            except RuntimeError as e:
                print(e)
            models[i][j] = model
            training_times[i][j] = training_time

    # model prediction based on the previously saved test dataset elements
    for i in range(10):
        prediction_times[i] = {}
        output_prediction[i] = {}
        for j in range(10):
            try:
                start_time = time.time()
                output_prediction[i][j] = models[i][j].predict(Dataframes_xtest[i][j])
                prediction_time = time.time() - start_time
            except RuntimeError as e:
                print(e)
            prediction_times[i][j] = prediction_time
            


    # calculation of model metrics
    for i in range(10):
        accuracies[i] = {}
        precisions[i] = {}
        recalls[i] = {}
        f1[i] = {}
        for j in range(10):
            test_split = train_size[j]
            accuracies[i][j] = accuracy_score(output_prediction[i][j],Dataframes_ytest[i][j])
            precisions[i][j] = precision_score(output_prediction[i][j],Dataframes_ytest[i][j],average='weighted',zero_division=1)
            recalls[i][j] = recall_score(output_prediction[i][j],Dataframes_ytest[i][j],average='weighted',zero_division=1)
            f1[i][j] = f1_score(output_prediction[i][j],Dataframes_ytest[i][j],average='weighted',zero_division=1)
    
    # Finding the best accuracy
    best_acc = 0
    best_acc_model = None
    best_acc_ratio = None
    for i in accuracies:
        for j in accuracies[i]:
            test_split = train_size[j]
            if accuracies[i][j] > best_acc:
                best_acc = accuracies[i][j]
                best_acc_model = i + 1
                best_acc_ratio = 1 - test_split  # training set ratio

    print(f"Best Accuracy: {best_acc:.4f} (Model {best_acc_model}, Training set ratio {best_acc_ratio:.2f})")

    # Finding the best precision
    best_pre = 0
    best_pre_model = None
    best_pre_ratio = None
    for i in precisions:
        for j in precisions[i]:
            test_split = train_size[j]
            if precisions[i][j] > best_pre:
                best_pre = precisions[i][j]
                best_pre_model = i + 1
                best_pre_ratio = 1 - test_split  # training set ratio

    print(f"Best Precision: {best_pre:.4f} (Model {best_pre_model}, Training set ratio {best_pre_ratio:.2f})")

    # finding the best recall score
    best_rec = 0
    best_rec_model = None
    best_rec_ratio = None
    for i in recalls:
        for j in recalls[i]:
            test_split = train_size[j]
            if recalls[i][j] > best_rec:
                best_rec = recalls[i][j]
                best_rec_model = i + 1
                best_rec_ratio = 1-test_split
    print(f"Best Recall: {best_rec:.4f} (Model {best_rec_model}, Training set ratio {best_rec_ratio:.2f})")

    # finding the best f1-score
    best_f1 = 0
    best_f1_model = None
    best_f1_ratio = None
    for i in f1:
        for j in f1[i]:
            test_split = train_size[j]
            if f1[i][j] > best_f1:
                best_f1 = f1[i][j]
                best_f1_model = i + 1
                best_f1_ratio = 1-test_split
    print(f"Best F1-score: {best_f1:.4f} (Model {best_f1_model}, Training set ratio {best_f1_ratio:.2f})")

    # calculating the model that required the shortest training time
    min_time = 100 # using a large value relative to the expected time for comparison purposes
    min_time_model = None
    min_time_ratio = None
    for i in training_times:
        test_split = train_size[j]
        for j in training_times[i]:
            if training_times[i][j] < min_time:
                min_time = training_times[i][j]
                min_time_model = i+1
                min_time_ratio = 1 - test_split
    print(f"Shortest time {min_time} (Model {min_time_model}, Training set Ratio {min_time_ratio:.2f})")


    # finding the mean values for each metric to create arrays for plotting
    for j in range(10):
        # mean accuracy for each model split
        avg_acc = np.mean([accuracies[i][j] for i in range(10)])
        mean_acc.append(avg_acc)
        std_acc.append(np.std([accuracies[i][j] for i in range(10)]))
        # mean precision for each model split
        avg_pre = np.mean([precisions[i][j] for i in range(10)])
        mean_pre.append(avg_pre)
        std_pre.append(np.std([precisions[i][j] for i in range(10)]))
        # mean recall for each model split
        avg_rec = np.mean([recalls[i][j] for i in range(10)])
        mean_rec.append(avg_rec)
        std_rec.append(np.std([recalls[i][j] for i in range(10)]))
        # mean f1-score for each model split
        avg_f1 = np.mean([f1[i][j] for i in range(10)])
        mean_f1.append(avg_f1) 
        std_f1.append(np.std([f1[i][j] for i in range(10)]))

    # plotting the various datas
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    train_ratios = 1 - train_size
    # Accuracy
    axs[0, 0].plot(train_ratios, mean_acc, marker='o', color='blue',label='Mean Accuracy')
    axs[0,0].fill_between(train_ratios,np.array(mean_acc) - np.array(std_acc),
                          np.array(mean_acc) + np.array(std_acc),
                          color='blue',alpha=0.1,label='Std Dev')
    axs[0, 0].set_title("Learning Curve (Accuracy)")
    axs[0, 0].set_xlabel("Fraction of Training Data")
    axs[0, 0].set_ylabel("Accuracy")
    axs[0, 0].grid(True)
    axs[0,0].legend()
    # Recall
    axs[1, 0].plot(train_ratios, mean_rec, marker='o', color='orange',label='Mean Recall')
    axs[1,0].fill_between(train_ratios,np.array(mean_rec) - np.array(std_rec),
                          np.array(mean_rec) + np.array(std_rec),color='orange',alpha=0.1,label='Std Dev')
    axs[1, 0].set_title("Recall Score Curve")
    axs[1, 0].set_xlabel("Training Set Ratio")
    axs[1, 0].set_ylabel("Recall")
    axs[1, 0].grid(True)
    axs[1,0].legend()
    # F1-score
    axs[1, 1].plot(train_ratios, mean_f1, marker='o', color='red',label='Mean F1-Score')
    axs[1,1].fill_between(train_ratios,np.array(mean_f1) - np.array(std_f1),
                          np.array(mean_f1) + np.array(std_f1),color='red',alpha=0.1,label='Std Dev')
    axs[1, 1].set_title("F1 Score Curve")
    axs[1, 1].set_xlabel("Training Set Ratio")
    axs[1, 1].set_ylabel("F1 Score")
    axs[1, 1].grid(True)
    axs[1,1].legend() 
    # Precision
    axs[0, 1].plot(train_ratios, mean_pre, marker='o', color='green',label='Mean Precision')
    axs[0,1].fill_between(train_ratios,np.array(mean_pre) - np.array(std_pre),
                          np.array(mean_pre) + np.array(std_pre),color='green',alpha=0.1,label='Std Dev')
    axs[0, 1].set_title("Precision Score Curve")
    axs[0, 1].set_xlabel("Training Set Ratio")
    axs[0, 1].set_ylabel("Precision")
    axs[0, 1].grid(True)
    axs[0,1].legend()

    plt.tight_layout()
    plt.savefig("Classifier_Metrics.png")
    plt.show()



    # returning the required values
    return {
        accuracies,
        precisions,
        recalls,
        f1
    }


if __name__ == '__main__':
    classifier()