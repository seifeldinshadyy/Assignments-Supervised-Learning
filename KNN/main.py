import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from KNN import KNN

# Argument Parsing 
parser = argparse.ArgumentParser(description="K-Nearest Neighbors algorithm with selectable distance metric.")
parser.add_argument("--k", type=int, default=3, help="Number of neighbors (default: 3)")
parser.add_argument("--distance", type=str, choices=["euclidean", "cosine", "manhattan"], default="euclidean",
                    help="Distance metric to use (default: euclidean)")

args = parser.parse_args()



df = pd.read_csv('Iris.csv')
df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]



print("First 5 rows of dataset:")
print(df.head())

# Plotting the data
sns.lmplot(x="sepal_width", y="sepal_length", hue="species", data=df, fit_reg=False)
plt.title("Sepal Width vs Sepal Length")
plt.show()

sns.lmplot(x="petal_width", y="petal_length", hue="species", data=df, fit_reg=False)
plt.title("Petal Width vs Petal Length")
plt.show()



X = df.drop(columns=["species"]).values
y = df["species"].values


# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=15)



# Train and Test KNN Classifier
clf = KNN(k=args.k, distance_metric=args.distance)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print("Predictions:")
print(predictions)

# myAccuracy 
myaccuracy = np.sum(predictions == y_test) / len(y_test)
# Accuracy Calculation
accuracy = accuracy_score(y_test, predictions)
correct_predictions = np.sum(predictions == y_test)
incorrect_predictions = len(y_test) - correct_predictions

print(f"Chosen Distance Metric: {args.distance}")
print(f"Number of Neighbors (k): {args.k}")
print(f"My Accuracy: {myaccuracy}")
print(f"Accuracy: {accuracy}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Incorrect Predictions: {incorrect_predictions}")