import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree


df = pd.read_csv("loan.csv")
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

genderMapping = {
    "Male": 0,
    "Female": 1
}
marriedMapping = {
    "No": 0,
    "Yes": 1
}
educationMapping = {
    "Not Graduate": 0,
    "Graduate": 1
}
selfEmployedMapping = {
    "No": 0,
    "Yes": 1
}
propertyAreaMapping = {
    "Rural": 0,
    "Urban": 1,
    "Semiurban": 2
}
loanStatusMapping = {
    "N": 0,
    "Y": 1
}

df.replace(to_replace=r"3\+", value='4', regex=True, inplace=True)
df.replace(to_replace=r"LP", value='0', regex=True, inplace=True)
df["Gender"] = df["Gender"].map(genderMapping)
df["Married"] = df["Married"].map(marriedMapping)
df["Education"] = df["Education"].map(educationMapping)
df["Self_Employed"] = df["Self_Employed"].map(selfEmployedMapping)
df["Property_Area"] = df["Property_Area"].map(propertyAreaMapping)
df["Loan_Status"] = df["Loan_Status"].map(loanStatusMapping)

X = df.drop(columns=["Loan_Status"])
X = X.drop(columns=["Loan_ID"])
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

output = pd.DataFrame(X_test)
output.insert(11, 'Predictions', predictions)

genderout = {
    0: 'Male',
    1: 'Female'
}
marriedout = {
    0: 'No',
    1: 'Yes'
}
educationout = {
    0: "Not Graduate",
    1: "Graduate"
}
selfEmployedout = {
    0: "No",
    1: "Yes"
}
propertyAreaout = {
    0: "Rural",
    1: "Urban",
    2: "Semiurban"
}
loanStatusout = {
    0: "No",
    1: "Yes"
}

output["Gender"] = output["Gender"].map(genderout)
output["Married"] = output["Married"].map(marriedout)
output["Education"] = output["Education"].map(educationout)
output["Self_Employed"] = output["Self_Employed"].map(selfEmployedout)
output["Property_Area"] = output["Property_Area"].map(propertyAreaout)
output["Predictions"] = output["Predictions"].map(loanStatusout)

print(output.to_string())

tree.export_graphviz(model, out_file='visualizedTree.dot',
                     feature_names=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area'],
                     class_names=sorted(str(y.unique())),
                     label='all',
                     rounded=True,
                     filled=True)
