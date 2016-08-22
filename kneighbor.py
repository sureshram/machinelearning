
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
y = iris.target # f(X) = y (so lowercase)

from sklearn.cross_validation import train_test_split
# partition the data into two sets, X_train, y_train for training and X_test, y_test for testing

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)


