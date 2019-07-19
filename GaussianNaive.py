

import numpy as np
import pandas as pd

from google.colab import files
up = files.upload()

import io
df = pd.read_csv(io.BytesIO(up['train.csv']),encoding = 'unicode_escape',parse_dates=['Date(ET)'])

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stops = stopwords.words('english')

from nltk.stem.porter import PorterStemmer
corpusBody = []
for i in range(0, 1157):
    dummy =  df['TRANS_CONV_TEXT'][i]
    diagnosis = re.sub('[^a-zA-Z]', ' ',str(dummy))
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split()
    ps = PorterStemmer()
    diagnosis = [ps.stem(word) for word in diagnosis if not word in set(stops)]
    diagnosis = ' '.join(diagnosis)
    corpusBody.append(diagnosis)

from nltk.stem.porter import PorterStemmer
corpusTitle = []
for i in range(0, 1157):
    dummy =  df['Title'][i]
    diagnosis = re.sub('[^a-zA-Z]', ' ',str(dummy))
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split()
    ps = PorterStemmer()
    diagnosis = [ps.stem(word) for word in diagnosis if not word in set(stops)]
    diagnosis = ' '.join(diagnosis)
    corpusTitle.append(diagnosis)

badaCorpus = []
for i,j in zip(corpusTitle,corpusBody):
  badaCorpus.append(i+' '+j)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 1500)
cpB = cv.fit_transform(badaCorpus).toarray()

cpBdataFrame=pd.DataFrame(cpB)

cpBdataFrame['Source']=df['Source']

cpBdataFrame['Source'].value_counts()

for i in range(0,1157):
  if cpBdataFrame['Source'][i] == 'Facebook':
    cpBdataFrame['Source'][i] = 'FACEBOOK'

cpBdataFrame=pd.get_dummies(cpBdataFrame,drop_first=True)

cpBdataFrame.head(1)

cpBdataFrame['year'] = df['Date(ET)'].dt.year
cpBdataFrame['month'] = df['Date(ET)'].dt.month
cpBdataFrame['day'] = df['Date(ET)'].dt.day
y=df.iloc[:,-1]

from sklearn.model_selection import train_test_split
qtrain,qtest,ytrain,ytest = train_test_split(cpBdataFrame,y,test_size=0.3,stratify = y,random_state=2019)

from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(qtrain,ytrain)
predy=gnb.predict(qtest)
from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,predy))

"""------------------------------------------TEST FILE-------------------------------------------------------------"""



import io
df_test = pd.read_csv('test.csv',encoding = 'unicode_escape',parse_dates=['Date(ET)'])

TestBody = []
for i in range(0, 571):
    dummy =  df_test['TRANS_CONV_TEXT'][i]
    diagnosis = re.sub('[^a-zA-Z]', ' ',str(dummy))
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split()
    ps = PorterStemmer()
    diagnosis = [ps.stem(word) for word in diagnosis if not word in set(stops)]
    diagnosis = ' '.join(diagnosis)
    TestBody.append(diagnosis)

TestTitle = []
for i in range(0, 571):
    dummy =  df['Title'][i]
    diagnosis = re.sub('[^a-zA-Z]', ' ',str(dummy))
    diagnosis = diagnosis.lower()
    diagnosis = diagnosis.split()
    ps = PorterStemmer()
    diagnosis = [ps.stem(word) for word in diagnosis if not word in set(stops)]
    diagnosis = ' '.join(diagnosis)
    TestTitle.append(diagnosis)

TestCorpus = []
for i,j in zip(TestTitle,TestBody):
  TestCorpus.append(i+' '+j)

cv_test = TfidfVectorizer(max_features = 1500)
cpB_test = cv_test.fit_transform(TestCorpus).toarray()



cpBtestFrame=pd.DataFrame(cpB_test)
cpBtestFrame['Source']=df_test['Source']
for i in range(0,571):
  if cpBtestFrame['Source'][i] == 'Facebook':
    cpBtestFrame['Source'][i] = 'FACEBOOK'
cpBtestFrame=pd.get_dummies(cpBtestFrame,drop_first=True)
df_test['Date(ET)'] = pd.to_datetime(df_test['Date(ET)'], errors='coerce')
cpBtestFrame['year'] = df_test['Date(ET)'].dt.year
cpBtestFrame['month'] = df_test['Date(ET)'].dt.month
cpBtestFrame['day'] = df_test['Date(ET)'].dt.day

from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='most_frequent')
df_Imputed = imp.fit_transform(cpBtestFrame)

cpBtestFrame = pd.DataFrame(df_Imputed,columns= cpBtestFrame.columns)

predy=gnb.predict(cpBtestFrame)

predy

