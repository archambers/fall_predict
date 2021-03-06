"""

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
import mpld3

#create dataframe from csv file
df = pd.read_csv('Fixed_Flat_Surface.csv', sep=',')

#populate features array
build_features = []
for i in range(len(df.SbjID)):
	build_features.append([
		df.Age[i], 
		#df.A_NormCad_Avg[i], 
		#df.A_SWPperc_Avg[i], 
		#df.A_SWPperc_CV[i],
		#df.A_SL_CV[i], 
		df.A_SpeedNorm_Avg[i], 
		#df.A_NormTstr_Avg[i]
	])


features_train = np.array(build_features[:])
features_train[:,0] /= features_train[:,0].max()
features_train[:,1] /= features_train[:,1].max()
#features_train[:,2] /= features_train[:,2].max()
'''
features_test = np.array(build_features[40:50])
features_test[:,0] /= features_test[:,0].max()
features_test[:,1] /= features_test[:,1].max()
#features_test[:,2] /= features_test[:,2].max()
'''


#populate labels array
build_labels = []
for i in range(len(df.SbjID)):
	build_labels.append(
		df.Falls[i]
		)

labels_train = np.array(build_labels[:])
#labels_test = np.array(build_labels[40:50])

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
#clf = SVC(C=.1, kernel='linear')
#clf = KNeighborsClassifier()
#clf = DecisionTreeClassifier()
#clf = AdaBoostClassifier()
#clf = RandomForestClassifier()

clf.fit(features_train, labels_train)
'''
pred = []
for i in features_test:
	pred.append(clf.predict(i))


predictions = np.array(pred)
'''
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
'''
from sklearn.metrics import accuracy_score

print str(np.ravel(predictions)) + " : predictions"
print str(labels_test) + " : true"

print(accuracy_score(labels_test,predictions))
'''

#print features_train[:,0]


import matplotlib.pyplot as plt
import pylab as pl

x_min = features_train[:,0].min() - .05; x_max = features_train[:,0].max() + .05
y_min = features_train[:,1].min() - .05; y_max = features_train[:,1].max() + .05

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
h = .001  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())


fig, ax = plt.subplots()

cont = ax.contourf(xx, yy , Z, zorder=-1)
#plt.pcolormesh(xx, yy, Z, cmap=pl.cm.seismic)


vAge = features_train[:,0]
vSpeed = features_train[:,1]

#vAge2 = features_test[:,0]
#vSpeed2 = features_test[:,1]

#plt.scatter(vAge2, vSpeed2, c=labels_test, marker='v')
scatter = ax.scatter(vAge, vSpeed, c=labels_train, zorder=-1)


#for i in range(len(features_train)):
#	plt.annotate(str(df.SbjID[i]), xy=(features_train[:,0][i],features_train[:,1][i]), bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 0.5))

labels = [str(df.SbjID[i]) for i in range(len(features_train))]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()
#plt.show()
