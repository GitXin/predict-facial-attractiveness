import numpy as np
from sklearn import decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

import pdb

features = np.loadtxt('../data/train_features', delimiter=',')
ratings = np.loadtxt('../data/train_ratings', delimiter=',')
pca = decomposition.PCA(n_components=20)
regr = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)

for i in range(0, 500):
  features_train = np.delete(features, i, 0)
  features_test = features[i, :]
  pca.fit(features_train)
  features_train = pca.transform(features_train)
  features_test = pca.transform(features_test.reshape(1, -1))
  ratings_train = np.delete(ratings, i, 0)
  regr.fit(features_train, ratings_train)
  print('number of models trained:', i + 1)

pca.fit(features)
features = pca.transform(features)
ratings_predict = regr.predict(features)
corr = np.corrcoef(ratings_predict, ratings)[0, 1]
print('Correlation:', corr)
joblib.dump(regr, '../model/my_face_rating.pkl', compress=1)

print("Generate Model Successfully!")
