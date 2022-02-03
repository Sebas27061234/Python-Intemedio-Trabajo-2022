"""Descripcion de que modelo se ha elegido"""
"""Describir funciones"""
import pandas as pd
import seaborn as sns
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import GradientBoostingClassifier as SGD
from sklearn.preprocessing import MinMaxScaler as MMS
from IPython.display import display
ruta = 'WineQT.csv'
wine_df= pd.read_csv("WineQT.csv")
wine_df = wine_df.drop(columns='Id')



def modelPreprocess():
    global quality
    global X , y
    bins = (1,5.5,10)
    group_names = ['bad','good']
    wine_df['quality'] = pd.cut(wine_df['quality'], bins = bins, labels = group_names)
    quality = wine_df
    label_quality = LabelEncoder()
    wine_df['quality'] = label_quality.fit_transform(wine_df['quality'])
    wine_df['quality'].value_counts()
    X = np.array(wine_df.drop('quality', axis = 1))
    y = np.array(wine_df['quality'])
    escaler = StandardScaler()
    X = escaler.fit_transform(X)
    MaxScaler = MMS()
    X = MaxScaler.fit_transform(X)

def ModelMachineLearning():
    Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size = 0.2, random_state = 42)

    ModeloSGD_wine = SGD(n_estimators=100,max_depth = 7,max_features=5,loss = 'exponential',criterion ='friedman_mse', random_state=42).fit(Xtrain, ytrain)

    ypredtest = ModeloSGD_wine.predict(Xtest)
    ypredtrain = ModeloSGD_wine.predict(Xtrain)
    accuracyModel = input('Desea ver la precision que tiene el modelo? (S/N): ')
    if accuracyModel == 'S':
        x = ModeloSGD_wine.score(Xtrain, ytrain)
        w = ModeloSGD_wine.score(Xtest, ytest)
        w = round(w,4)*100
        print('La precision del modelo para determinar si un vino es Bueno o Malo \nEs de: {}%'.format(w))
        Respuesta = True
    elif accuracyModel == 'N':
        Respuesta = True
    else:
        print('La accion que esta realizando es incorrecta\n Reiniciando programa')
        Respuesta = False
    return Respuesta



def initiation():
    print('Bienvenido al programa de prediccion de vinos.')
    modelPreprocess()
    query = input('Ingrese "P", para evaluar si su Vino es Bueno o Malo: ')
    if query == 'P':
        Respuesta = ModelMachineLearning()
        if Respuesta == True:
            modelTest()
        elif Respuesta == False:
            initiation()
    else:
        print('Cerrando seccion.....')



modelPreprocess()
ModelMachineLearning()
