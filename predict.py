import joblib
from sklearn.datasets import load_iris

model = joblib.load("iris_model.pkl")

iris = load_iris()

print("Enter Flower Details")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

prediction = model.predict([[sl, sw, pl, pw]])

print("Predicted Flower:",
      iris.target_names[prediction[0]])