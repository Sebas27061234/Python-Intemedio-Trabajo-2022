import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import MinMaxScaler as MMS
from IPython.display import display
ruta = 'https://raw.githubusercontent.com/Sebas27061234/Python-Intemedio-Trabajo-2022/main/WineQT.csv'
wine_df= pd.read_csv(ruta)
wine_df = wine_df.drop(columns='Id')

def modelTest():
    x1 = wine_df.drop(columns = ['quality'])
    x1 = list(x1.columns)
    list1 = []
    for i in x1:
        pred_i = input('Ingresa el valor {}: '.format(i))
        list1.append(pred_i)
    if list1 != []:
        respuesta_pred = ModeloRFC.predict([list1])
        return respuesta_pred
    elif list1 == []:
        print('No ha ingresado ningun dato....Retornando')
        initiation()

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
    global ModeloRFC
    Xtrain, Xtest, ytrain, ytest = tts(X, y, test_size = 0.2, random_state = 42)
    ModeloRFC = RFC(n_estimators=55,max_depth = None,max_features=5,random_state=42)
    ModeloRFC.fit(Xtrain,ytrain)
    accuracyModel = input('Desea ver la precision que tiene el modelo? (S/N): ')
    if accuracyModel == 'S':
        scoretest = ModeloRFC.score(Xtest, ytest)
        print(f'La presicion de este modelo es de: {round(scoretest*100,3)}%')
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
    query = input('Ingrese "P", para evaluar si su Vino es Bueno o Malo\nEnter para salir\n: ')
    if query == 'P':
        Respuesta = ModelMachineLearning()
        if Respuesta == True:
            respuest = modelTest()
            if respuest == [0]:
                print('El vino es de baja calidad')
            if respuest == [1]:
                print('El vino es de buena calidad')
        elif Respuesta == False:
            initiation()
    else:
        print('Cerrando seccion.....')


initiation()