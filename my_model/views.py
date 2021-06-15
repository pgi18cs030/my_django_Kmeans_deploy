from django.http import HttpResponse
from django.shortcuts import render
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def index(request):
    return render(request,'index.html')

def prediction_model(l1):
    data = pd.read_csv('Social_Network_Ads.csv')
    X = data.iloc[:, [2, 3]].values
    sc = StandardScaler()
    X = sc.fit_transform(X)
    pred = np.array(l1)
    print(pred)
    pred = pred.reshape(1, 2)
    print(pred)
    model = pickle.load(open('my_model.pkl', 'rb'))
    return model.predict(sc.transform(pred))


def result(request):
    l1=[]
    Age=int(request.GET.get('age'))
    Salary =int(request.GET.get('salary'))
    l1.append(Age)
    l1.append(Salary)
    output=prediction_model(l1)
    print(output)
    if output == [0]:
        prediction = "Customer is careless"

    elif output == [1]:
        prediction = "Customer is standard"
    elif output == [2]:
        prediction = "Customer is Target"
    elif output == [3]:
        prediction = "Customer is careful"

    else:
        prediction = "Custmor is sensible"

    print(prediction)
    passs = {'prediction_text': 'Model has Predicted', 'output': prediction}
    return render(request,'result.html',passs)