import streamlit as st
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

st.write('Penguins')
st.markdown('企鵝數據探索分析') 

pengu = pd.read_csv('penguins.csv')
#st.write(pengu.head(10))

selected_species = st.selectbox('要看什麼物種？', ['Adelie', 'Gentoo', 'Chinstrap'])
pengu_df = pengu[pengu['species'] == selected_species]

selected_x_var = st.selectbox('X 軸是什麼？', 
  ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']) 
selected_y_var = st.selectbox('Y 軸是什麼？', 
  ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g']) 

# graph each species by hue and shape
sns.set_style('darkgrid')
markers = {"Adelie": "X", "Gentoo": "s", "Chinstrap":'o'}

fig, ax = plt.subplots()

ax = sns.scatterplot(data = pengu_df, x = selected_x_var, 
  y = selected_y_var, hue = 'species', markers = markers,
  style = 'species') 

plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title("scatterplot")
st.pyplot(fig)