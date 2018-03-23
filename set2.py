import pandas as pd
from scipy.io import arff

data = arff.loadarff('TrainingDataset.arff')
filedata = pd.DataFrame(data[0])

X = filedata.drop(['Result'],axis=1)
Y = filedata.filter(['Result'],axis=1)
Y_arr = Y.values.ravel()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, Y_arr, test_size=0.33, random_state=42)


print 'DECISION TREE'
from sklearn import tree,svm
clf = tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
print clf.score(X_test,y_test)


print 'RBF SVM'
clf1 = svm.SVC(decision_function_shape='ovo',kernel='rbf')
clf1.fit(X_train,y_train)
print clf1.score(X_test,y_test)


print 'RANDOM FOREST'
from sklearn import ensemble
clf2 = ensemble.RandomForestClassifier()
clf2.fit(X_train,y_train)
print clf2.score(X_test,y_test)


print 'ADBOOST'
from sklearn import ensemble
clf3 = ensemble.AdaBoostClassifier()
clf3.fit(X_train,y_train)
print clf3.score(X_test,y_test)


print 'KNN'
from sklearn import neighbors
clf4 = neighbors.KNeighborsClassifier(11)
clf4.fit(X_train,y_train)
print clf4.score(X_test,y_test)


print 'NAIVE BAYES'
from sklearn import naive_bayes
clf5 = naive_bayes.GaussianNB()
clf5.fit(X_train,y_train)
print clf5.score(X_test,y_test)


print 'Tree using entropy'
clf_entropy = tree.DecisionTreeClassifier(criterion = "entropy")
clf_entropy.fit(X_train, y_train)
clf_entropy.fit(X_train,y_train)
print clf_entropy.score(X_test,y_test)


print 'MLP'
from sklearn import neural_network
clf6 = neural_network.MLPClassifier(alpha=1,max_iter=600,solver='lbfgs',shuffle=True)
clf6.fit(X_train,y_train)
print clf6.score(X_test,y_test)
