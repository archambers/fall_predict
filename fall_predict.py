"""
Classifiers to use:
	GaussianNB
	SVC
	K-Nearest Neighbors
	Decistion Tree/ Forest

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
for i in range(len(df.SbjID)-1):
	build_features.append([
		df.Age[i], 
		df.A_NormCad_Avg[i], 
		df.A_SWPperc_Avg[i], 
		df.A_SWPperc_CV[i], 
		df.A_SpeedNorm_Avg[i], 
		df.A_NormTstr_Avg[i]
	])

features_train = np.array(build_features)

#populate labels array
build_labels = []
for i in range(len(df.SbjID)-1):
	build_labels.append([
		df.Falls[i]
		])

labels_train = np.array(build_labels)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
print(clf.predict([
	df.Age[4], 
	df.A_NormCad_Avg[4], 
	df.A_SWPperc_Avg[4], 
	df.A_SWPperc_CV[4], 
	df.A_SpeedNorm_Avg[4],
	df.A_NormTstr_Avg[4]
]))
