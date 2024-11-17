from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

iris = load_iris()
X = pd.DataFrame(iris.data, columns= iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

dogruluk = accuracy_score(y_test, y_pred)
print("Model Doğruluk Oranı:", dogruluk)
print("\nSınflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
plt.imshow(cm, interpolation= "nearest", cmap= plt.cm.Blues)
plt.title("Karışıklık Matrisi")
plt.colorbar()
plt.xticks([0,1,2], iris.target_names)
plt.yticks([0,1,2], iris.target_names)
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], horizontalalignment = "center", color = "white" if cm[i, j] > cm.max() / 2 else "black")
plt.show()

ozellik_onemleri = pd.Series(model.feature_importances_, index = iris.feature_names)
plt.figure(figsize= (8,6))
ozellik_onemleri.sort_values().plot(kind = "barh", color = "skyblue")
plt.title("Özellik Önemleri")
plt.xlabel("Önem")
plt.ylabel("Özellikler")
plt.show()
