import numpy as np
import scipy as sp
import pandas as pd
import sklearn
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
import sklearn.model_selection as model_selection
from sklearn.metrics import classification_report,roc_curve,auc,accuracy_score,roc_auc_score,confusion_matrix
from sklearn.model_selection import learning_curve 
from sklearn.tree import DecisionTreeClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

df=pd.read_csv('data.csv') 

# PCA MDS
from sklearn.decomposition import PCA
X=df.values[:,1:13]
y=df.values[:,13]
pca = PCA(n_components=7)
X_pca = pca.fit_transform(X)
mds = MDS(n_components=2, dissimilarity='euclidean')
X_mds = mds.fit_transform(X_pca)
print(pca.explained_variance_ratio_)  
plt.scatter(X_mds[y == 0, 0], X_mds[y == 0, 1], c='red', marker='o', label='HLB')
plt.scatter(X_mds[y == 1, 0], X_mds[y == 1, 1], c='blue', marker='x', label='Asymptomatic')
plt.scatter(X_mds[y == 2, 0], X_mds[y == 2, 1], c='green', marker='+', label='Health')
plt.xlabel('Compoment 1 (70.8%)')
plt.ylabel('Compoment 2 (16%)')
plt.legend()
plt.title('MDS PCA')
plt.show()


# RF model parameters 
X=df.values[:,1:13]
y=df.values[:,13]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
rfc = RandomForestClassifier(n_estimators=68, max_depth=9, min_samples_split=5, min_samples_leaf=2,random_state=42)
rfc = rfc.fit(xtrain,ytrain)

#RF-MDS
proba = rfc.predict_proba(X)
mds = MDS(n_components=5)
mds_result = mds.fit_transform(proba)
plt.scatter(mds_result[y == 0, 0], mds_result[y == 0, 1], c='red', marker='o', label='HLB')
plt.scatter(mds_result[y == 1, 0], mds_result[y == 1, 1], c='blue',marker='x', label='Asymptomatic')
plt.scatter(mds_result[y == 2, 0], mds_result[y == 2, 1], c='green', marker='+', label='Health')
plt.scatter(mds_result[:, 0], mds_result[:, 1], c=y)
plt.title('MDS HLB-RF')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
plt.legend()

# Average score
for i in range(4,11,1):
clf = RandomForestClassifier(n_estimators=47, max_depth=none, min_samples_split=2, min_samples_leaf=1,random_state=42)
scores = cross_val_score(clf, xtest, ytest,cv=5)
print(scores.mean())
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#OOB data
rfc = RandomForestClassifier(n_estimators=68,oob_score=True)
rfc = rfc.fit(X,y)
print(rfc.oob_score_)

#Print ROC curve
y2=np.array(y,dtype=int)
color=['r*-','b*-','g*-']
P=rfc.predict_proba(X)
print(y2)
print(rfc.predict_proba(X))
for k in set(y2):
label = []
for i in y2:
if i==k:label.append(1)
else:label.append(0)
score=[P[a,k] for a in range(P.shape[0])]
print(label)
print(score)
FPR,TPR,yuzhi=roc_curve(label,score,pos_label=1)
AUC=auc(FPR,TPR)
plt.plot(FPR,TPR,color[k],label='class'+str(k)+' AUC='+str('%.3f'%AUC),linewidth=2)
plt.xlim([-0.05, 1])
plt.ylim([-0.00, 1.05])
plt.xlabel('Specitificity')
plt.ylabel('Sensitivitys')
plt.plot([0,1],[0,1],'--',label='45°',linewidth=2)
plt.legend()
plt.show()

#Print Micro-average ROC curve
L=np.zeros(shape=(P.shape[0],P.shape[1]))
for i in range(P.shape[0]):L[i,y2[i]]=1
FPR,TPR,yuzhi=roc_curve(L.ravel(),P.ravel())
AUC=auc(FPR,TPR)
plt.xlim([-0.01, 1])
plt.ylim([-0.01, 1.01])
plt.plot(FPR,TPR,'o-',color='blue',label='microROC AUC='+str('%.3f'%AUC),linewidth=2)
plt.plot([0,1],[0,1],'--',label='45°',linewidth=2)
plt.legend()
plt.show()



# Scoring Curve
for i in range(3,11,1):
plt.plot(range(1,i+1),cross_val_score(rfc,xtrain,ytrain,cv=i),label='train')
plt.plot(range(1,i+1),cross_val_score(rfc,xtest,ytest,cv=i),label='test')
plt.legend()
plt.show()
print('rfc prediction:',rfc.predict(xtest))
print(ytest)

# Learning Curve
fig,ax=plt.subplots(1,1,figsize=(6,6))
train_sizes,train_scores,test_scores=learning_curve(rfc,X,y,n_jobs=1,cv=6)
ax.set_ylim((0.2,1.1))
ax.set_xlabel("training examples") ax.set_ylabel("score")
ax.grid()
ax.plot(train_sizes,np.mean(train_scores,axis=1),'o-',color='r',label='train score')
ax.plot(train_sizes,np.mean(test_scores,axis=1),'o-',color='g',label='test score')
ax.legend(loc='best')
plt.show()


# Feature_importances
print (rfc.feature_importances_)
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]# Print the feature ranking
print("Feature ranking:")
# f, ax = plt.subplots(figsize=(7, 5))
# ax.bar(range(len(rfc.feature_importances_)), rfc.feature_importances_)
# ax.set_title("Feature Importances")
# plt.show()

# Confusion_matrix
y_pred=rfc.predict(xtest)
matrix = confusion_matrix(ytest,y_pred,labels=None, sample_weight=None)
plt.imshow(matrix, cmap=plt.cm.Greens)
indices = range(len(matrix))
classes = range(0,3)
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
for first_index in range(len(matrix)):
for second_index in range(len(matrix[first_index])):
plt.text(first_index, second_index, matrix[first_index][second_index])
plt.show()

# Optimization parameters
rfc = RandomForestClassifier(n_estimators=42)
rfc_s = cross_val_score(rfc,X,y,cv=5)
re = []
for i in range(80):
rfc = RandomForestClassifier(n_estimators=i+1)
rfc_s = cross_val_score(rfc,X,y,cv = 5).mean()
re.append(rfc_s)
print(max(re),re.index(max(re)))
plt.plot(range(1,81),re)
plt.show()
import time
start = time.time()
param_grid = {'n_estimators':np.arange(30, 80),'max_depth':np.arange(1, 12),
'min_samples_leaf':np.arange(1, 9),'min_samples_split':np.arange(2, 9),}
rfc = RandomForestClassifier(random_state=42)
GS = GridSearchCV(rfc,param_grid,cv=5)
GS.fit(xtrain,ytrain)
end = time.time()
print(end-start)
print(GS.best_params_)
print(GS.best_score_)
