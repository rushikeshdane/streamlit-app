import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pandas_profiling import ProfileReport
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso ,RidgeCV,LassoCV , ElasticNet , ElasticNetCV,LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

from streamlit_pandas_profiling import st_profile_report

def main():
 #   """ Common ML Dataset Explorer """,
    st.title("Air Temperature Prediction ")
    st.image('./ai.jpg')
    st.subheader("2020 Predictive Maintenance Dataset")

    df = pd.read_csv('ai4i2020 (1).csv')  # loading dataset

    if st.checkbox("Show Dataset"):
        number = st.number_input("Number of Rows to View",5,100)
        st.dataframe(df.head(number))
    EDA = st.sidebar.selectbox('EDA', ['pandas profile report','correlation of feature','selected features','statistics'])

    if EDA =='correlation of feature':
     #   if st.sidebar.checkbox('correalation of features'):
            def corr(type,cmap='Blues'):
                fig, ax = plt.subplots()
                sns.heatmap(df.corr(type), ax=ax, cmap=cmap)
                return st.write(fig)
            corr_var= st.sidebar.selectbox("select correalation", ['pearson', 'kendall', 'spearman'])

            if corr_var == 'kendall':
                corr('kendall',cmap="Greens")
            if corr_var == 'pearson':
                corr('pearson')
            if corr_var  == 'spearman':
                corr('spearman',cmap="BuPu")
    profile = ProfileReport(df)

#    if st.sidebar.button('pandas profile report'):
    if EDA == 'pandas profile report':
        if st.sidebar.checkbox('view pandas profile report'):
                st_profile_report(profile)

    X = df.loc[:, ['UDI', 'HDF', 'Process temperature [K]']]

    Y = df['Air temperature [K]']

    if EDA == 'selected features':
       # if st.sidebar.checkbox('selected features for machine learning model'):
            st.title('selected features for machine learning model')
    #   for i in range(0,2):
            cols = st.columns(2)
            cols[0].write('X columns')
            cols[1].write('target variable')
            cols[0].write(X)
            cols[1].write(Y)


    if EDA =='statistics':
            st.title('Statistics for selected features')
      #  if st.sidebar.checkbox('statas for given data'):
            import statsmodels.formula.api as smf
            lm = smf.ols(formula='Y ~ X', data=df).fit()
            st.write(lm.summary())

    scaler = StandardScaler()
    arr = scaler.fit_transform(X)


    x_train, x_test, y_train, y_test = train_test_split(arr, Y, test_size=0.15, random_state=45)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    test1 = scaler.transform([[9993, 0, 308.4]])
    lassocv = LassoCV(alphas=None, cv=100, max_iter=50000, normalize=True)
    lassocv.fit(x_train, y_train)

    ridgecv = RidgeCV(alphas=np.random.uniform(0, 10, 50), cv=10, normalize=True)
    ridgecv.fit(x_train, y_train)

    elastic = ElasticNetCV(alphas=None, cv=10)
    elastic.fit(x_train, y_train)


    ml_model = st.sidebar.selectbox('select macahine learning model',['linear','lasso','ridge','elasitc'])
    def predictor(model):
            #model = lr
            model.fit(x_train,y_train)
            test = scaler.transform(([[a,b,c]]))
            return model.predict(test)


    a = st.sidebar.slider('UDI',1,10000)
    b = st.sidebar.slider('HDF',0,1)
    c = st.sidebar.slider('Process Temparature',305.7,313.8)


    if ml_model == 'linear':


        if st.button('Predict Air Temperature '):

            st.write('the air temperature for given machine learning model is {}'.format(predictor(lr)))

    elif ml_model == 'lasso':
        if st.button('Predict Air Temperature'):
         #   st.write('the air temparature for your inpute data is {}'.format(predictor(lassocv)))
            st.write('the air temperature for given machine learning model is {}'.format(predictor(lassocv)))

    elif ml_model == 'ridge':
        if st.button('Predict Air Temperature '):
            st.write('the air temperature for given machine learning model is {}'.format(predictor(ridgecv)))

    else :
        if st.button('Predict Air Temperature '):
            st.write('the air temperature for given machine learning model is {}'.format(predictor(elastic)))

    def predictor(model):
           # model = lr
            model.fit(x_train,y_train)
            test = scaler.transform(([[a,b,c]]))
            return model.predict(test)


if __name__ == '__main__':
	main()