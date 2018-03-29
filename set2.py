import pandas as pd
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt
data = arff.loadarff('TrainingDataset.arff')
filedata = pd.DataFrame(data[0])

X = filedata.drop(['Result'],axis=1)
Y = filedata.filter(['Result'],axis=1)
Y_arr = Y.values.ravel()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, Y_arr, test_size=0.33, random_state=42)



def binarize(x):
    if(int(x)==1):
        return 1
    else:
        return 0


def classifier_metrics(y_true,y_pred):
    y_pred = [binarize(x) for x in y_pred]
    y_test = [binarize(x) for x in y_true]
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

print 'DECISION TREE'
from sklearn import tree,svm,metrics
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
print clf.score(X_test,y_test)


print 'RBF SVM'
clf1 = svm.SVC(decision_function_shape='ovo',kernel='rbf')
clf1.fit(X_train,y_train)
print clf1.score(X_test,y_test)
classifier_metrics(clf1.predict(X_test),y_test)


print 'RANDOM FOREST'
from sklearn import ensemble
clf2 = ensemble.RandomForestClassifier()
clf2.fit(X_train,y_train)
print clf2.score(X_test,y_test)
classifier_metrics(clf2.predict(X_test),y_test)


print 'ADABOOST'
from sklearn import ensemble
clf3 = ensemble.AdaBoostClassifier()
clf3.fit(X_train,y_train)
print clf3.score(X_test,y_test)
classifier_metrics(clf3.predict(X_test),y_test)

print 'KNN'
from sklearn import neighbors
clf4 = neighbors.KNeighborsClassifier(11)
clf4.fit(X_train,y_train)
print clf4.score(X_test,y_test)
classifier_metrics(clf4.predict(X_test),y_test)

print 'NAIVE BAYES'
from sklearn import naive_bayes
clf5 = naive_bayes.GaussianNB()
clf5.fit(X_train,y_train)
print clf5.score(X_test,y_test)
classifier_metrics(clf5.predict(X_test),y_test)

print 'Tree using entropy'
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(X_train, y_train)
clf_entropy.fit(X_train,y_train)
print clf_entropy.score(X_test,y_test)
classifier_metrics(clf_entropy.predict(X_test),y_test)

print 'MLP'
from sklearn import neural_network
clf6 = neural_network.MLPClassifier(alpha=1,max_iter=600,solver='lbfgs',shuffle=True)
clf6.fit(X_train,y_train)
print clf6.score(X_test,y_test)
classifier_metrics(clf6.predict(X_test),y_test)
