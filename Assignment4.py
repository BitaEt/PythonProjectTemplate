import sys
import pandas as pd
import plotly.express as px
import statsmodels
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from plotly import graph_objects as go
import numpy
from plotly import figure_factory as ff
from sklearn.metrics import confusion_matrix
from itertools import combinations

df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
)
# print(df)

df.rename(columns={"class": "v_class"}, inplace=True)
# print(list(df))

# make a list of predictor and response variables
responses = ["survived"]
predictors = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked",
    "v_class",
    "who",
    "adult_male",
    "deck",
    "embark_town",
    "alive",
    "alone",
]

# print(list(responses))
# print(list(predictors))

# getting rid of missed values ASAP
for col in df.columns:
    if df[col].dtypes == "float":
        df[col].fillna((df[col].mean()), inplace=True)
    else:
        df = df.apply(lambda col: col.fillna(col.value_counts().index[0]))

# determine if boolean or continuous

boolean_labels = {}
for i in responses:
    if df[i].nunique() == 2:
        df[i] = df[i].astype("bool")
        df.replace({False: 0, True: 1}, inplace=True)
        boolean_labels = {idx: value for idx, value in enumerate(df[i].unique())}
# print(boolean_labels)

# create dictionary for predictors type
predictors_type = {"continuous": [], "categorical": []}
continuous = df.select_dtypes(include=["float"])

for i in predictors:
    if i in list(continuous):
        predictors_type["continuous"].append(i)
    else:
        predictors_type["categorical"].append(i)
print(predictors_type)

data_cat = df.select_dtypes("object")

# one hot encoder for categorical variables
onehotencoder = OneHotEncoder(handle_unknown="ignore")
encoder = onehotencoder.fit_transform(data_cat.values.reshape(-1, 1)).toarray()
dfOneHot = pd.DataFrame(encoder)
data = pd.concat([df.select_dtypes(exclude=["object"]), dfOneHot], axis=1)
data = data.head(len(df))
# print(data)

# creating train and test set
for i in responses:
    x = data.drop(i, axis=1)
    y = data[i]
# print(x)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

# I created a dictionary because of the sake of interpretation but lists are easier to work with in this plot example.

predictors_con = []
predictors_cat = []
for i in predictors:
    if i in list(continuous):
        predictors_con.append(i)
    else:
        predictors_cat.append(i)


# Plotting the variables

def cont_resp_cat_predictor():

    n = 200

    # Add histogram data and group data together
    for i in predictors:
        if i in predictors_cat and len(predictors_cat) > 2:
            hist_data = list(combinations(predictors_cat, 3))
        elif i in predictors_cat and 0 < len(predictors_cat) < 3:
            hist_data = list(combinations(predictors_cat, len(predictors_cat)))

    for j in df.columns:
        if j in responses and response_type == 'continuous':
            group_labels = sorted(df[i].unique())

    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Response",
        yaxis_title="Distribution",
    )
    fig_1.show()

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Response by Categorical Predictor",
        xaxis_title="Groupings",
        yaxis_title="Response",
    )
    fig_2.show()

    return


def cat_resp_cont_predictor():
    n = 200

    # Add histogram data
    hist_data = []
    for i in predictors:
        if i in predictors_con and len(predictors_con) > 2:
            hist_data = list(combinations(predictors_con, 3))
        elif i in predictors_cat and 0 < len(predictors_con) < 3:
            hist_data = list(combinations(predictors_con, len(predictors_con)))

    for i in df.columns:
        if i in responses and response_type == 'categorical':
            group_labels = sorted(df[i].unique())


    # Create distribution plot with custom bin_size
    fig_1 = ff.create_distplot(hist_data, group_labels, bin_size=0.2)
    fig_1.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Predictor",
        yaxis_title="Distribution",
    )
    fig_1.show()

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=numpy.repeat(curr_group, n),
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )
    fig_2.update_layout(
        title="Continuous Predictor by Categorical Response",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_2.show()

    return

def cat_response_cat_predictor():
    n = 200

    x = []
    for i in predictors:
        if i in predictors_cat and len(predictors_cat) > 2:
            x = list(*combinations(predictors_cat, 3))
        elif i in predictors_cat and 0 < len(predictors_cat) < 3:
            x = list(*combinations(predictors_cat, len(predictors_cat)))

    for i in df.columns:
        if i in responses and response_type == 'categorical':
            y = sorted(df[i].unique())

    x_2 = [1 if abs(x_) > 0.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 0.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (without relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()


    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig_no_relationship = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )
    fig_no_relationship.update_layout(
        title="Categorical Predictor by Categorical Response (with relationship)",
        xaxis_title="Response",
        yaxis_title="Predictor",
    )
    fig_no_relationship.show()

    return


def cont_response_cont_predictor():
    n = 200
    x = []
    for i in predictors:
        if i in predictors_con and len(predictors_con) > 2:
            x = list(*combinations(predictors_con, 3))
        elif i in predictors_con and 0 < len(predictors_con) < 3:
            x = list(*combinations(predictors_con, len(predictors_con)))

    for i in df.columns:
        if i in responses and response_type == 'categorical':
            y = sorted(df[i].unique())

    fig = px.scatter(x=x, y=y, trendline="ols")
    fig.update_layout(
        title="Continuous Response by Continuous Predictor",
        xaxis_title="Predictor",
        yaxis_title="Response",
    )
    fig.show()

    return


# creating models based on response category
for i in responses:
    if df[i].nunique() == 2:
        model = LogisticRegression()
        model.fit(x_train, y_train)
        predictor_reg = model.predict(x_test)
    else:
        feature_name = i
        predictor = statsmodels.api.add_constant(x[i].values)
        model = statsmodels.api.OLS(y, predictor)
        model_fitted = model.fit()
        print(f"Variable: {feature_name}")
        print(model_fitted.summary())

        # T-value and P-value
        t_value = round(model_fitted.tvalues[1], 6)
        p_value = "{:.6e}".format(model_fitted.pvalues[1])

        # Plot
        fig = px.scatter(x=[i].values, y=y, trendline="ols")
        fig.update_layout(
            title=f"Variable: {feature_name}: (t-value={t_value}) (p-value={p_value})",
            xaxis_title=f"Variable: {feature_name}",
            yaxis_title="y",
        )
        fig.show()

        # Random forest
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(x_train, y_train)
        sorted_idx = rf.feature_importances_.argsort()
        plt.barh(x.feature_names[sorted_idx], rf.feature_importances_[sorted_idx])
        plt.xlabel("Random Forest Feature Importance")
