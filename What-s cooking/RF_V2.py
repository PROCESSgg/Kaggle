import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.
df_train = pd.read_json('../input/train.json')
df_test = pd.read_json('../input/test.json')

#利用词袋对特征进行处理
df_train_ing = df_train.ingredients
df_test_ing = df_test.ingredients

word_train = [' '.join(x) for x in df_train_ing]
word_test = [' '.join(x) for x in df_test_ing]
vectorizer = CountVectorizer(max_features=1000)
bag_of_words = vectorizer.fit(word_train)
ing_array_train = bag_of_words.transform(word_train).toarray()
ing_array_test = bag_of_words.transform(word_test).toarray()
pd_ing_train = pd.DataFrame(ing_array_train, columns=vectorizer.vocabulary_)
pd_ing_test = pd.DataFrame(ing_array_test, columns=vectorizer.vocabulary_)

df_train_new = df_train.merge(pd_ing_train, left_index=True, right_index=True).drop('ingredients', axis=1)
df_test_new = df_test.merge(pd_ing_test, left_index=True, right_index=True).drop('ingredients', axis=1)

X_train = df_train_new.drop(['id','cuisine'], axis=1)
Y_train = df_train_new.cuisine
X_test = df_test_new.drop(['id'], axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, train_size=0.85)

RF = RandomForestClassifier(n_estimators=300, oob_score=True)
RF.fit(X_train,Y_train)

Y_test = RF.predict(X_test)
df_output = pd.DataFrame(np.array([df_test.id, Y_test]).T, columns=['id','cuisine']).set_index('id')
df_output.to_csv('output_RF.csv')
