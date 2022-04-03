from sklearn import svm
import numpy as np
from sklearn.inspection import permutation_importance
import sklearn.model_selection as model_selection
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import pickle

with open('../output/features/features_biotech.pkl', 'rb') as file:
    features = pickle.load(file)

two_authors = features.loc[features['id'].isin([691951, 2373875])]

X = two_authors.drop(['id'], axis = 1).values.tolist()
y = two_authors['id'].tolist()
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80, test_size=0.20)

rbf = svm.SVC(kernel='rbf', C=2).fit(X_train, y_train)
perm_importance = permutation_importance(rbf, X_test, y_test)

rbf_pred = rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_pred)
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
print('Accuracy (RBF Kernel): ', "%.2f" % (rbf_accuracy*100))
print('F1 (RBF Kernel): ', "%.2f" % (rbf_f1*100))

feature_names = ['yule', 'fk_grade', 'f_reading', 'gunning_fog', 'honore_r', 'avg_word_length', 'syllable_no', 'spelling_errors', 'no_tag', 'sym', 'punct', 'mean_word_rank', 'of_freq', 'is_freq', 'the_freq', 'been_freq','nvn','nnv','vnn','vnv','nap','nnc','nad','dna','nnn','nan','vad']
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")

kernels = ['Polynomial', 'RBF', 'Sigmoid','Linear']
def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return svm.SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return svm.SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return svm.SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return svm.SVC(kernel='linear', gamma="auto")

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.20)
for i in range(4):    
    svclassifier = getClassifier(i) 
    svclassifier.fit(X_train, y_train)# Make prediction
    y_pred = svclassifier.predict(X_test)# Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test,y_pred))

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train,y_train)
print(grid.best_estimator_)
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))#Output