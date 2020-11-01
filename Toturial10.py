# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
import pandas
import re
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# url = "car.csv"
# names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'acc']
rawdata = pandas.read_csv(url, names=names)
array = rawdata.values
nrow, ncol = rawdata.shape
X = array[:, 0:8]
Y = array[:, 8]


# generate model and get accuracy

# def get_accuracy(target_train, target_test, predicted_test,predicted_train):
#     clf = DecisionTreeClassifier(max_depth=2)
#     clf.fit(predicted_train, np.ravel(target_train, order='C'))
#     predictions = clf.predict(predicted_test)
# #     predictions1 = clf.predict_proba(predicted_test)
# #     print(predictions1)
#     return accuracy_score(target_test, predictions) #0.72262774
# pred_train, pred_test, tar_train, tar_test = train_test_split(X, Y, test_size=.3, random_state=2)
# print("Accuracy score of our DCT model without feature selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,
#                                                                                     pred_train))
def get_accuracy1(target_train1, target_test1, predicted_test1,predicted_train1):
    clf1 = MLPClassifier(hidden_layer_sizes = (5,15),max_iter=150)
    clf1.fit(predicted_train1, np.ravel(target_train1, order='C'))
    predictions = clf1.predict(predicted_test1)
    # y_pred = clf1.predict(pred_test)
    # from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    # print(confusion_matrix(tar_test, y_pred))
    # print(classification_report(tar_test, y_pred))
    # print(accuracy_score(tar_test, y_pred))
    # predictions1 = clf.predict_proba(predicted_test)
    # print(predictions1)
    return accuracy_score(target_test1, predictions)

pred_train1, pred_test1, tar_train1, tar_test1 = train_test_split(X, Y, test_size=.3, random_state=2)
print("Accuracy score of our MLP model without feature selection : %.2f" % get_accuracy1(tar_train1, tar_test1, pred_test1,
                                                                                    pred_train1))

# from sklearn.ensemble import VotingClassifier
# voting_clf = VotingClassifier(estimators=[
#     ('DCT_clf', DecisionTreeClassifier(max_depth=2)),
#     ('MLP_clf', MLPClassifier(hidden_layer_sizes = (5,15))),
# ], voting='soft')
# voting_clf.fit(pred_train, tar_train)
# voting_clf.score(pred_test, tar_test)
# print(voting_clf.score(pred_test, tar_test))
# # feature extraction
# test = SelectKBest(score_func=chi2, k=7)# K:Number of top features to select.
# fit = test.fit(X, Y)
# # summarize scores
# np.set_printoptions(precision=3)
# print(fit.scores_)
# features = fit.transform(X)
# # summarize selected features
# print(features[0:4, :],"summerize features")
# print()
#
# # Now apply only the K most significant features according to the chi square method
# pred_features = features[:, 0:3]
# pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
# print("Accuracy score of our model with chi square feature selection : %.2f" % get_accuracy(tar_train, tar_test,
#                                                                                             pred_test,pred_train))
# print()
# from sklearn.feature_selection import SelectFromModel
# # Feature Extraction with RFE
# model = LogisticRegression()  # Logistic regression is the Wrapper classifier here
# rfe = RFE(model, 3)
# fit = rfe.fit(X, Y)
# #print("Num Features: %d" % (fit.n_features_))
# #print("Selected Features: %s" % (fit.support_))
# #print("Feature Ranking: %s" % (fit.ranking_))
# ##Now apply only the K most significant features according to the RFE feature selection method
# features = fit.transform(X)
# pred_features = features[:, 0:3]
# #
# pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
# print("Accuracy score of our model with RFE selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
# print()
#
# ## Feature Extraction with PCA
#
#
# ## feature extraction
# pca = PCA(n_components=3)
# fit = pca.fit(X)
# features = fit.transform(X)
# ## summarize components
# #rint("Explained Variance: %s" % (fit.explained_variance_ratio_))
# #print(fit.components_)
# #
# ##Now apply only the K most significant faetures (components) according to the PCA feature selection method
# #features = fit.transform(X)
# pred_features = features[:, 0:3]
# pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
# print("Accuracy score of our model with PCA selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,pred_train))
# print()
# #
# ## Feature Importance with Extra Trees Classifier
#
# from sklearn.ensemble import ExtraTreesClassifier
#
# ## feature extraction
# model = ExtraTreesClassifier(max_depth=3,min_samples_leaf=2)
# fit = model.fit(X, Y)
# print(model.feature_importances_)
# print()
# t = SelectFromModel(fit, prefit=True)
# features = t.transform(X)
# pred_features = features[:, 0:3]
#
# pred_train, pred_test, tar_train, tar_test = train_test_split(pred_features, Y, test_size=.3, random_state=2)
# print("Accuracy score of our model with Extra Trees selection : %.2f" % get_accuracy(tar_train, tar_test, pred_test,
#                                                                                      pred_train))
# #print()