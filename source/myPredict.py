from sklearn.externals import joblib
import numpy as np
from sklearn import decomposition

clf = joblib.load('../model/my_face_rating.pkl')
features = np.loadtxt('../data/train_features', delimiter=',')
my_features = np.loadtxt('../data/my_features', delimiter=',')
if my_features.ndim == 1: my_features = np.reshape(my_features, (1, -1))
amount = len(my_features)
pca = decomposition.PCA(n_components=20)
pca.fit(features)

predictions = np.zeros([amount,1]);

for i in range(0, amount):
  features_test = my_features[i, :]
  features_test = pca.transform([features_test])
  predictions[i] = clf.predict(features_test)

print(predictions)
