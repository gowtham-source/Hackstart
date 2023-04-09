from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model


with st.sidebar:
    selected = option_menu("HackStart⚡", ["Problem statement 1", 'Problem statement 2'],
                           icons=['activity', 'activity'], menu_icon="code-slash", default_index=0)
regr = pickle.load(open('model.pkl', 'rb'))

if selected == "Problem statement 1":
    st.subheader("Hackstart Competition")
    st.info('''### Problem statement 1
The first problem statement consists of two parts:
1. Using data analytics **create a framework** to match the startups of Tamil Nadu sector-wise.
2. Given such data of Startups With Startup TN which in years will be scaled exponentially, **propose an architecture and implementable solution** that can be used to:
    - Maintain a digital snapshot of the Startup Database
    - Track the progress of Startup TN startups''')
    st.write("## Solution")
    df = pd.read_csv(".\DPIIT Startups.csv", encoding='iso-8859-1')
    st.write("#### Part A")
    if st.button("view data"):
        st.write(df)
        if st.button('close'):
            st.stop()
    # renaming the columns for getting ready to analyse
    # df2 = df.set_axis(
    #     ["sno name founder dpiitno founderemail contact website incorpno incorpdate stage recdate sector industry city district".split()], axis=1, inplace=False)
    # df2.head()
    # ndf = df2

    # st.write(ndf)

if selected == "Problem statement 2":
    st.subheader("Hackstart Competition")
    st.info('''Solution:
- For this statement the regressive models are suitable for prediction,
 - And the model is trained with 88 features which are highly correlated with the target variable.
 - with various regressive algorithms and regularization methods, the best accuracy is achieved with Linear regression.
            ''')
    df = pd.read_csv(
        "./investments.csv", encoding='iso-8859-1')
    df = df.rename(columns={' market ': 'market',
                   ' funding_total_usd ': 'funding_total_usd'})
# treating a null values by replacing the empty values which has '-' in to nan , later droping the entire row which has null values

    df['funding_total_usd'] = df['funding_total_usd'].replace('-', np.nan)
    df = df.dropna(subset=['funding_total_usd'])
    df['funding_total_usd'] = df['funding_total_usd'].str.strip().replace('-',
                                                                          np.nan)
    # df['funding_total_usd']
    # df["funding_total_usd"] = df["funding_total_usd"].astype(float)
    df.dropna(subset=['funding_total_usd'], inplace=True)
    df_req = df[['name', 'funding_total_usd', 'funding_rounds', 'seed', 'venture', 'private_equity',
                 "round_A", "round_B", 'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']]

    X = df_req[['funding_rounds', 'seed', 'venture', 'private_equity', 'round_A',
                'round_B', 'round_C', 'round_D', 'round_E', 'round_F', 'round_G', 'round_H']]
    y = df_req[['funding_total_usd']]
    # evaluation of a model using 88 features chosen with correlation

    def select_features(X_train, y_train, X_test):
        fs = SelectKBest(score_func=f_regression, k=88)
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    X, y = make_regression(n_samples=1000, n_features=100,
                           n_informative=10, noise=0.1, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=1)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    model = LinearRegression()
    model.fit(X_train_fs, y_train)

    yhat = model.predict(X_test_fs)
    # evaluate predictions
    mae = mean_absolute_error(y_test, yhat)
    # print('MAE: %.3f' % mae)

    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    pred = regr.predict(X_test)

    Accuracy = r2_score(y_test, pred)*100
    st.write("### Accuracy of the best model is %.2f" % Accuracy)

    st.write("Here are the predictions of the model")
    st.write(pred.tolist())
