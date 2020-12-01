# import required libraries

import streamlit as st
import numpy as np
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from PIL import Image

# disable warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

# set title
st.title('DevOps Team presents')

# import image
image = Image.open('versicolor.jpg')
st.image(image)

# set subtitle
st.write("""
    # A simple ML app with Streamlit
""")

st.write("""
    # Lets Explore different classification with Iris dataset
""")

# choosing dataset in sidebar
dataset_name = st.sidebar.selectbox(
    'Select dataset: ', ('Iris', ''))

# choosing classifier for dataset in sidebar
classifier_name = st.sidebar.selectbox(
    'Select classifier: ', ('SVM', 'KNN', 'Decision Tree', 'Random Forest'))

# function to get dataset


def get_dataset(name):
    data = None
    if name == 'Iris':
        data = datasets.load_iris()

    elif name == 'Wine':
        data = datasets.load_wine()

    else:
        data = datasets.load_breast_cancer()

    x = data.data
    y = data.target

    return x, y


x, y = get_dataset(dataset_name)
st.dataframe(x)
st.write('Shape of your datast is: ', x.shape)
st.write('The unique data: ', len(np.unique(y)))

# visulaization
fig = plt.figure()
sns.boxplot(data=x, orient='h')
st.pyplot()

plt.hist(x)
st.pyplot()

# Building our algorithm


def add_parametr(name_of_clf):
    params = dict()
    if name_of_clf == 'SVM':
        c = st.sidebar.slider('C', 0.01, 15.0)
        g = st.sidebar.slider('G', 0.01, 15.0)
        params['C'] = c
        params['G'] = g
    elif name_of_clf == 'KNN':
        k = st.sidebar.slider('K', 1, 15)
        params['K'] = k
    return params


params = add_parametr(classifier_name)


#  access classifer function
def get_classifier(name_of_clf, params):
    clf = None
    if name_of_clf == 'SVM':
        clf = SVC(C=params['C'], gamma=params['G'])

    elif name_of_clf == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])

    elif name_of_clf == 'Decision Tree':
        clf = tree.DecisionTreeClassifier(criterion='gini')

    elif name_of_clf == 'Random Forest':
        clf = RandomForestClassifier()
    return clf


clf = get_classifier(classifier_name, params)


# train and test data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10)

clf.fit(x_train, y_train)  # 80% for training
y_pred = clf.predict(x_test)  # 20% for testing
st.write(y_pred)

accuracy = accuracy_score(y_test, y_pred)
st.write('classifier name: ', classifier_name)
st.write('Accuracy score for your model is: ', accuracy)
st.balloons()
