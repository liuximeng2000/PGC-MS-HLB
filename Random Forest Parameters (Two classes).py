import numpy as np
import scipy as sp
import pandas as pd
import sklearn
from numpy import indices
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,  GridSearchCV, cross_val_score, learning_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from itertools import cycle
import matplotlib as mpl
from scipy import interp
import seaborn as sns


df=pd.read_csv('E:\\A-data\\python-4\\RF-D.W.csv',encoding='utf-8') 

X=df.iloc[:,1:13] 
y=df.iloc[:,13]

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3,random_state=42)

#RF class
rfc = RandomForestClassifier(n_estimators=26,max_depth=3,min_samples_split=3,min_samples_leaf=1,random_state=42)
rfc = rfc.fit(xtrain,ytrain)
roc_auc_score
result1 = roc_auc_score(ytest, rfc.predict_proba(xtest)[:, 1])   #TRAIN
result2 = roc_auc_score(ytrain, rfc.predict_proba(xtrain)[:, 1])   #TEST
print(result1)
print(result2)
# Accuracy
y_pred = rfc.predict(xtest)
print(accuracy_score(ytest, y_pred))
print(accuracy_score(ytest, y_pred, normalize=False))

#SCORE
for i in range(3,11,1):
plt.plot(range(1,i+1),cross_val_score(rfc,xtrain,ytrain,cv=i),label='train')
plt.plot(range(1,i+1),cross_val_score(rfc,xtest,ytest,cv=i),label='test')
plt.legend()
plt.show()
print('rfc prediction:',rfc.predict(xtest))
print(ytest)


#LEARN CORVE
fig,ax=plt.subplots(1,1,figsize=(6,6)) 
train_sizes,train_scores,test_scores=learning_curve(rfc,X,y,n_jobs=1,cv=5) 
ax.set_ylim((0.1,1.1)) 
ax.set_xlabel("training examples") 
ax.set_ylabel("score")
ax.grid() 
ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label='train score')
ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label='test score')
ax.legend(loc='best')
plt.show()


#TMPORTANT FEATURE
print (rfc.feature_importances_)
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]# Print the feature ranking
print("Feature ranking:")

#Debugging parameter
RF = RandomForestClassifier(random_state = 1)
score = cross_val_score(RF,xtrain,ytrain,cv=5).mean()
print(score)
RF = RandomForestClassifier(criterion = 'entropy',random_state = 1)
score = cross_val_score(RF,xtrain,ytrain,cv=5).mean()
print(score)

for i in range(3,10,1):
param_test1 = {'n_estimators': range(5,80,1)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=2, random_state=42),
                        param_grid = param_test1,
                        scoring='roc_auc',
                        cv=5)
gsearch1.fit(xtrain, ytrain)
print(gsearch1.best_params_, gsearch1.best_score_)
print()

param_test2 = {'max_depth':range(1, 10, 1)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=26,
                                                           min_samples_split=2,
                                                           random_state=42),
                       param_grid = param_test2,

                       scoring='roc_auc',
                        cv=5)
gsearch2.fit(xtrain,ytrain)
print(gsearch2.best_params_, gsearch2.best_score_)
roc_auc_score(ytest, gsearch2.best_estimator_.predict_proba(xtest)[:,1])
gsearch2.best_estimator_

param_test3 = {'min_samples_split':range(2, 10, 1), 'min_samples_leaf':range(1, 10, 1)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=26,
                                                            max_depth=3, random_state=42),
                        param_grid = param_test3,
                         scoring='roc_auc',
                        cv=5)
gsearch3.fit(xtrain,ytrain)
print(gsearch3.best_params_, gsearch3.best_score_)

param_grid = { 'max_features':np.arange(0.1, 1)}
rfc = RandomForestClassifier(random_state=1,n_estimators = 5,max_depth = 3,min_samples_leaf =1 ,min_samples_split =2 )
GS = GridSearchCV(rfc,param_grid,cv=7)
 GS.fit(xtrain,ytrain)
 print(GS.best_params_)
 print(GS.best_score_)


param_test5 = {'max_features':range(1, 12, 1)}
 gsearch5 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=18,min_samples_leaf =3 ,min_samples_split =2,
                                                            max_depth=2, random_state=1),
                                                           param_grid = param_test5,
                                                            scoring='roc_auc',
                                                            cv=5)
gsearch5.fit(xtrain,ytrain)
 print(gsearch5.best_params_, gsearch5.best_score_)




# ROC
predictions_validation = rfc.predict_proba(xtest)[:,1]
from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(ytest, predictions_validation)
roc_auc = auc(fpr, tpr)
plt.title('ROC Validation')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc,linewidth=3)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.01, 1])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Confusion matrix
y_pred=rfc.predict(xtest)
cm= confusion_matrix(ytest, y_pred,labels=[0,1])
sklearn.metrics.confusion_matrix(ytest,y_pred, labels=[0,1], sample_weight=None)
labels=[0,1]
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
plt.title('confusion matrix')  
plt.xlabel('Predict lable') 
plt.ylabel('True lable') 
plt.show()


#score
clf = RandomForestClassifier(n_estimators=18, max_depth=3,min_samples_split=2,max_features=3, min_samples_leaf=1,random_state=22)
scores = cross_val_score(clf, xtest, ytest,cv=6)
print(scores.mean())
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
