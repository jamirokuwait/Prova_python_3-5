import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlem


def main():
    
    new_model = mlem.api.load('model_.mlem')
    df = pd.read_csv('formart_house.csv') 
    df = df.drop([506])
    df = df.rename(columns={'medv': 'Price'})
    df= df.apply(pd.to_numeric)
    
    df['chas'] = df['chas'].astype(float)
    df['rad'] = df['rad'].astype(float)
    df['tax'] = df['tax'].astype(float)

    st.dataframe(df)
    tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8,tab9,tab10,tab11,tab12,tab13 =st.tabs(['crim','zn','indus','chas','nox','rm','age','dis','rad','tax','ptratio','b','lstat'])
    with tab1:
        crim = st.number_input('crim',df['crim'].min(),df['crim'].max(),df['crim'].mean())
    with tab2:
        zn = st.number_input('zn',df['zn'].min(),df['zn'].max(),df['zn'].mean())
    with tab3:
        indus= st.number_input('indus',df['indus'].min(),df['indus'].max(),df['indus'].mean())
    with tab4:
        chas = st.number_input('chas',df['chas'].min(),df['chas'].max(),df['chas'].mean())
    with tab5:
        nox = st.number_input('nox',df['nox'].min(),df['nox'].max(),df['nox'].mean())
    with tab6:
        rm = st.number_input('rm',df['rm'].min(),df['rm'].max(),df['rm'].mean())
    with tab7:
        age = st.number_input('age',df['age'].min(),df['age'].max(),df['age'].mean())
    with tab8:
        dis = st.number_input('dis',df['dis'].min(),df['dis'].max(),df['dis'].mean())
    with tab9:
        rad = st.number_input('rad',df['rad'].min(),df['rad'].max(),df['rad'].mean())
    with tab10:
        tax = st.number_input('tax',df['tax'].min(),df['tax'].max(),df['tax'].mean())
    with tab11:
        ptratio = st.number_input('ptratio',df['ptratio'].min(),df['ptratio'].max(),df['ptratio'].mean())
    with tab12:
        b = st.number_input('b',df['b'].min(),df['b'].max(),df['b'].mean())
    with tab13:
        lstat = st.number_input('lstat',df['lstat'].min(),df['lstat'].max(),df['lstat'].mean())
    
    res = new_model.predict([[crim,zn,indus,chas,nox,rm,age,dis,rad,tax,ptratio,b,lstat]])[0]
    st.write('il nostro predict è:')
    st.write(round(res, 4))
    
    tab1,tab2=st.tabs(['Coeff_','Intercept_'])
    
    with tab1:
        st.write('Coeff:')
        st.write(new_model.coef_)
    with tab2:
        st.write('Intercept:')
        st.write(new_model.intercept_)
    
if __name__ == '__main__':
    main()