import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlem

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
   st.title('Prova sommativa 3/5')
    
    
   st.subheader('Plotting our dataset for viewing our data')
   df = pd.read_csv('formart_house.csv') 
   df = df.drop([506])
   df = df.rename(columns={'medv': 'Price'})
   
   df['chas'] = df['chas'].astype(float)
   df['rad'] = df['rad'].astype(float)
   df['tax'] = df['tax'].astype(float)

   st.dataframe(df)
   
   df= df.apply(pd.to_numeric)
   df.info()
   
   
   corr_df = df.corr()
   st.subheader('Heatmap')
   fig = plt.figure(figsize=(10, 8))
   sns.heatmap(corr_df,annot=True)
   st.pyplot(fig)
   
   st.subheader('Price countplot')
   fig = plt.figure(figsize=(10, 8))
   plt.hist(df.Price)
   plt.xlabel('Price')
   plt.title('Count Price')
   st.pyplot(fig)
   
   st.subheader('Correlation between Price and Crim')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,x = 'Price',y='crim',hue='age')
   plt.title('plot Price x Crim (hue = age)')
   st.pyplot(fig)
   
   st.subheader('Correlation between Price and Nox')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,x = 'Price',y='nox',hue='indus')
   plt.title('plot Price x Nox (hue = indus)')
   st.pyplot(fig)
   
   
   
   st.subheader('Correlation between Price and Age')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,x = 'Price',y='age',hue='tax')
   plt.title('plot Price x Age (hue = tax)')
   st.pyplot(fig)
   
   st.subheader('Correlation between Price and Rm')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,x = 'Price',y='rm',hue='lstat')
   plt.title('plot Price x Rm (hue = lstat)')
   st.pyplot(fig)
   
   st.subheader('Correlation between Price and Nox')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,x = 'age',y='nox',hue='Price')
   plt.title('plot Age x Nox (hue = Price)')
   st.pyplot(fig)
   
   st.subheader('Correlation between Indus and Dis')
   fig = plt.figure(figsize=(10, 8))
   sns.scatterplot(data = df,y = 'indus',x='dis',hue='nox')
   plt.title('plot indus x Dis (hue = Nox)')
   st.pyplot(fig)
   
#    grouprice= df.groupby('Price').mean()
   
   
   
   X = df.drop(columns=['Price'],axis = 1)
   y = df['Price']
   
   X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                                        test_size=0.3,
                                                                        random_state=667
                                                                        )
   model = LinearRegression()
   model.fit(X_train, y_train)
   y_pred = model.predict(X_test)
   length = y_pred.shape[0]
   x = np.linspace(0, length, length)
   
   res_df = pd.DataFrame(data=list(zip(y_pred, y_test)),columns=['predicted', 'real'])
   st.subheader('Real values of dataset and predicted values of our model')
   st.dataframe(res_df)
   
   r2score = r2_score(y_test, y_pred)
   mae = mean_absolute_error(y_test, y_pred)
   mse = mean_squared_error(y_test, y_pred)
   rmse = mean_squared_error(y_test, y_pred, squared=False)
   st.subheader('Various type of errors for our model')
   st.write('R2_score: ', r2score)
   st.write('MAE(mean absolute): ', mae)
   st.write('MSE(mean squared): ', mse)
   st.write('RMSE: ', rmse)

   
#    fig = plt.figure(figsize=(10, 8))
#    plt.scatter(x, y_test, label='real y')
#    st.pyplot(fig)
   st.subheader('Plot of our Values(Real and Predicted)')
   fig = plt.figure(figsize=(10, 8))
   plt.plot(x, y_test, label='real y')
   plt.plot(x, y_pred, '-r', label="predicted y'")
   plt.legend(loc=2)
   st.pyplot(fig)
   
   
   
   mlem.api.save(model,
              'model_', # model_.mlem
              sample_data = X_train #features
              )
   
   
if __name__ == '__main__':
    main()
