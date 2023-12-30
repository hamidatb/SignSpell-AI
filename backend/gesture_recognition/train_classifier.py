import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Getting the main file path and the file path of the data.pickle file (which stores the binary image dataset)
def get_directory():
    # Getting the directory where this script file is located
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    # Setting the directory where the data is stored, relative to this script file
    DATA_DIR = os.path.join(SCRIPT_DIR, "data.pickle")
    
    return SCRIPT_DIR, DATA_DIR

# Opening the pickle file
def open_pickle(SCRIPT_DIR, DATA_DIR):
    data_dict = pickle.load(open(DATA_DIR, "rb"))
    return data_dict

# Training using Sci-Kit learn in order to classify the hand symbols
def training(data_dict:dict):
    data = np.asarray(data_dict['data']) # data and labels are lists.
    labels = np.asarray(data_dict['labels'])

    # data = "x", labels = "y". Using the infor from the data and labels and splitting all the info into two different sets.
    # I am using 20% of the data to test, 80% to train (relatively standard). Its common practice to shuffle the data.
    # Straifying by the labels (0,1,2 or the respective letters I showed in each)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)

    # Checking the sccuracy score of the model
    score = accuracy_score(y_predict, y_test)
    print (f"{score*100}% of samples were classified correctly.")

    return model 

def save_model(SCRIPT_DIR, filename, model):
    full_path = os.path.join(SCRIPT_DIR, filename)
    with open(full_path, "wb") as file:
        pickle.dump({"model":model}, file)
        print(f"The new model file saved successfully to your current folder.")

def main():
    SCRIPT_DIR, DATA_DIR = get_directory()
    data_dict = open_pickle(SCRIPT_DIR, DATA_DIR)
    model = training(data_dict)
    save_model(SCRIPT_DIR, "model.p", model)

if __name__ == "__main__":
    main()