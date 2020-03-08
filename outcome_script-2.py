###########################
# Author: Michael Smith   #
# Chris Lucas,   		  #
# Ramin Ahmari (Jun 2018) #
###########################

import csv
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingRegressor, GradientBoostingClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier, LinearRegression, Ridge, LogisticRegression

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from sklearn.model_selection import cross_val_score


y = []
with open('/Users/MichaelSmith/Desktop/Code/recon/decision_labels_471.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	next(reader, None) # skip over header
	for row in reader: y.append(int(row[0]))

x = []
with open('/Users/MichaelSmith/Desktop/combined_binary_sans_docid.csv', 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	next(reader, None) # skip over header
	for row in reader: x.append([int(r) for r in row])

x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

classifiers = [RandomForestClassifier(n_estimators=10), \
				MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), \
				#BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5), \
				DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0), \
				#ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0), \
				#AdaBoostClassifier(n_estimators=100), \
				#GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0), \
				#GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls'), \
				LogisticRegression(random_state=1), \
				GaussianNB(), \
				KNeighborsClassifier(n_neighbors=7), \
				SVC(kernel='rbf', probability=True)]
				#SGDClassifier(loss="hinge", penalty="l2"), \
				#LinearRegression(), \
				#Ridge (alpha = .5)

predictions_dict = {}
cross_val_dict = {}
for classifier in classifiers:
	classifier.fit(X_test, y_test)
	#predictions_dict[classifier] = classification_report(y_test, classifier.predict(X_test))
	cross_val_dict[classifier] = cross_val_score(classifier, x, y, cv=5)

old_score = 0
best_key = ""
for key, value in predictions_dict.items():
	if old_score < value.mean():
		old_score = value.mean()
		best_key = key
	print("5-Fold Cross Validation Score")
	print(value.mean())
	print("Confidence Interval")
	print(value.std())
	print(value)

clf = classifiers[0]

importances = clf.feature_importances_
print(importances)
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
feature_names = ['apologize_comm', 'ask_agency_comm', 'give_agency_comm', 'gratitude_comm', 'please_comm', 'first_name_comm', 'last_name_comm', 'apologize_inm', 'ask_agency_inm', 'give_agency_inm', 'gratitude_inm', 'please_inm', 'Rude/Agressive_C', 'Emotive_C', 'Perceptive_C', 'Social_C', 'Friendly_C', 'Negative_C', 'Positive_C', 'Confrontational_C', 'Indecisive_C', 'Selfish_C', 'Rude/Agressive_I', 'Emotive_I', 'Perceptive_I', 'Social_I', 'Friendly_I', 'Negative_I', 'Positive_I', 'Confrontational_I', 'Indecisive_I', 'Selfish_I']

# Print the feature ranking
print("Feature Ranking:")

for f in range(x.shape[1]):
    print("%d. Feature %s (%f)" % (f + 1, feature_names[indices[f]], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()