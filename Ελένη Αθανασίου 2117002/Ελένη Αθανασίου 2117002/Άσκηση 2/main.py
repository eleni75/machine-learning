import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import seaborn as sns
from matplotlib import pyplot as plt


df = pd.read_csv('WineQT.csv')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

X = df.drop(columns=["Id", "quality"])
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

model = linear_model.LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

output = pd.DataFrame(X_test)
output.insert(11, 'Predictions', predictions)
print(output.to_string())

sns.set_theme(color_codes=True)
sns.regplot(x="Predictions", y="volatile acidity", data=output, scatter=True)
plt.show()
