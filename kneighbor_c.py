import random

'''
 k = # of nearest neighbors, euclidian distance - different b/w first set of features, diff b/w second set .. so on .... diff b/w n features
'''

class KNN():
  def fit(self, X_train, y_train):
   self.X_train = X_train
   self.y_train = y_train

  def predict(self, X_test):
   predictions = []
   for row in X_test:
      label = random.choice(y_train) # accuracy will be 34 percent
      predictions.append(label)
   return predictions 

from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target

print X, y

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)


