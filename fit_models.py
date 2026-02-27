import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler 
import pandas as pd
import joblib


#MLP Classifier
iterations = 1000
alpha = 3

# Random Forest Classifier
est = 6
depth = 5

# KNN
neighbors = 5

# Decision Tree
tree_depth = 7

random_state = 69

model_types = {
    'knn': KNeighborsClassifier(n_neighbors=neighbors), 
    'DT': DecisionTreeClassifier(min_samples_leaf = 5, max_depth = tree_depth), 
    'forest': RandomForestClassifier(n_estimators=est, max_depth = depth, random_state = random_state), 
    'mlp': MLPClassifier(max_iter= iterations, alpha= alpha, random_state = random_state), 
    'clf': LogisticRegression(random_state=random_state, C =1), 
    'gnb': GaussianNB(), 
    'svc': SVC(random_state = random_state, C = 1)
}


training_sets = ['master', 'ff', 'w1', 'w2', 'big', 'little', 'comp']

train_master = pd.read_csv("data/training/master.csv")

scaler = StandardScaler()
scaler.fit(train_master.drop(columns = ["TRAIN"]))
joblib.dump(scaler , 'my_scaler.pkl')

def scale(df):
    m = pd.DataFrame(scaler.transform(df), columns = df.columns)
    return m

def formatStuff(df):
    y = df["TRAIN"]
    x = scale(df.drop(columns = ["TRAIN"]))
    return (x,y)

for training_set in training_sets:
    df = pd.read_csv("data/training/" + training_set + ".csv")
    x_train, y_train = formatStuff(df)
    for model in model_types:
        model_types[model].fit(x_train, y_train)
        joblib.dump(model_types[model], 'models/' + model + "/" + training_set + ".pkl")
        print("Accuracy on {} {} training set: {:.3f}".format(model, training_set, model_types[model].score(x_train, y_train)))