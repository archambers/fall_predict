"""
Classifiers to use:
	GaussianNB
	SVC
	K-Nearest Neighbors
	Decision Tree/ Forest

Features to use:
	A_NormCad_Avg
	A_SWPperc_Avg
	A_NormTstr_Avg (Stride Time)
	A_SWPperc_CV
	A_SpeedNorm_Avg
	Age

"""



import numpy as np
import pandas as pd

#create dataframe from csv file
df = pd.read_csv('Fixed_Flat_Surface.csv', sep=',')

#populate features array
build_features = []
for i in range(len(df.SbjID)):
	build_features.append([
		df.Age[i], 
		df.A_NormCad_Avg[i], 
		df.A_SWPperc_Avg[i], 
		df.A_SWPperc_CV[i], 
		df.A_SpeedNorm_Avg[i], 
		df.A_NormTstr_Avg[i]
	])

features_train = np.array(build_features[:40]+build_features[50:])
features_test = np.array(build_features[40:50])

#populate labels array
build_labels = []
for i in range(len(df.SbjID)):
	build_labels.append(
		df.Falls[i]
		)

labels_train = np.array(build_labels[:40]+ build_labels[50:])
labels_test = np.array(build_labels[40:50])

'''
print(features_train)
print(labels_train)
'''


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

clf = GaussianNB()
#clf = SVC(kernel='rbf')
#clf = KNeighborsClassifier()
#clf = DecisionTreeClassifier()
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier()

clf.fit(features_train, labels_train)

pred = []
for i in features_test:
	pred.append(clf.predict(i))


predictions = np.array(pred)
'''
print(clf.predict([
	df.Age[74], 
	df.A_NormCad_Avg[74], 
	df.A_SWPperc_Avg[74], 
	df.A_SWPperc_CV[74], 
	df.A_SpeedNorm_Avg[74],
	df.A_NormTstr_Avg[74]
]))
'''
from sklearn.metrics import accuracy_score

print predictions
print labels_test

print(accuracy_score(labels_test,predictions))


