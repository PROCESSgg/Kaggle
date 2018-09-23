import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd

def read_json(path):
    with open(path) as f:
        data = json.load(f)
        # list of dishes: {id, cuisine, ingredients}
    return data
# data: [], {id:, cuisine:}
def cuisine(data):
    cuisines = []
    for d in data:
        cuisines.append(d['cuisine'])
    return cuisines

#def ingredient(data):
# {}
def tra_class(data):
    class_le = LabelEncoder()
    class_le.fit(data)
    y = class_le.transform(data)
    return y

def class_tra(data,y):
    class_le = LabelEncoder()
    class_le.fit(data)
    y = class_le.inverse_transform(y)
    return y

#返回原材料，格式[[],[],...]
def ingreadients(data):
    ingre = []
    for d in data:
        ingre.append(d['ingredients'])
    return ingre

def onehotself(ingredients, ingredient_dict):
    feature = np.zeros([len(ingredients), len(ingredient_dict)])
    for i in range(len(ingredients)):
        for j in ingredients[i]:
            feature[i][ingredient_dict[j]] = 1
    return feature

#生成材料的字典库
def generate_ingredients_dict(ingredients):
    ingredient_dict = {}

    for ing in ingredients:
        for i in ing:
            if i not in ingredient_dict:
                ingredient_dict[i] = len(ingredient_dict)
    return ingredient_dict

def list_to_csv(id, cuisines):
    columns = ['id', 'cuisine']
    dataframe = pd.DataFrame({'id':id, 'cuisine':cuisines},columns=columns)

    print(dataframe)
    dataframe.to_csv("sample_submission.csv", index=False)

def get_id(test_data):
    id = []
    for data in test_data:
        id.append(data['id'])
    return id

if __name__ =="__main__":
    #读取训练数据，列表型
    path_train = "all/train.json"
    path_test = "all/test.json"
    train_data = read_json(path_train)
    test_data = read_json(path_test)
    #提取cuisines,[[],[],[],...,]
    train_cuisines = cuisine(train_data)
    #分类字符型特征，转化为数字型标签，np.array
    train_y = tra_class(train_cuisines)
    #提取ingreadients，[[],[],...[]]
    train_ingre = ingreadients(train_data)
    test_ingre = ingreadients(test_data)
    #生成ingreadients的字典库，用于生成onehot向量
    ingre_dict = generate_ingredients_dict(train_ingre + test_ingre)
    #训练集特征和测试集特征
    train_features = onehotself(train_ingre, ingre_dict)
    test_features = onehotself(test_ingre, ingre_dict)
    test_id = get_id(test_data)

    #训练
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(train_features, train_y)
    print('end_train')
    test_y = class_tra(train_cuisines, rf.predict(test_features))
    print('end_test')
    list_to_csv(test_id, test_y)
    print('end')
