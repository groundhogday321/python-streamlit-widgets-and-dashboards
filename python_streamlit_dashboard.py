

 

# streamlit run python_streamlit_dashboard.py

# *** IMPORTS ***
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from vega_datasets import data as vds

# *** DATAFRAME ***
st.title('Cars DataFrame')
cars = vds.cars().dropna()
st.write(cars)

measures = ['Miles_per_Gallon', 'Cylinders', 'Displacement', 'Horsepower', 'Weight_in_lbs', 'Acceleration']

# *** SCATTER PLOT WITH DROPDOWN SELECTBOXES ***
st.title('Scatter Plot')
scatter_x = st.selectbox('x', measures)
scatter_y = st.selectbox('y', measures)
sns.regplot(x=scatter_x, y=scatter_y, data=cars)
st.pyplot()

# prediction model (i.e.-mpg at 7000 pounds)
st.title('Predict MPG')
weight = st.number_input('enter vehicle weight', min_value=1500, max_value=6000)
X = np.array(cars.Weight_in_lbs).reshape(-1,1)
y = np.array(cars.Miles_per_Gallon)
lr = LinearRegression().fit(X,y)
prediction = lr.predict(np.array([[weight]]))
st.text('predicted mpg')
st.text(f'{prediction[0]:.2f}')

# *** BOXPLOT WITH RADIO BUTTONS ***
st.title('Radio Buttons')
radio_button_options = st.radio('Choose measure for boxplot:', measures)

def create_boxplot(measure):
    swarmplot = st.checkbox('overlay swarmplot')
    if swarmplot and radio_button_options == measure:
        sns.boxplot(data=cars[measure])
        sns.stripplot(data=cars[measure], color='lightgrey')
        st.pyplot()
    else:
        sns.boxplot(data=cars[measure])
        st.pyplot()

create_boxplot(radio_button_options)

# show average mpg for Origin using some kind of widget