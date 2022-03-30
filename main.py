import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()


st.markdown(
    """
    <style>
    .main {
    background-color: #F5F5F5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache
def get_data(data):
	covid_data = pd.read_csv('https://raw.githubusercontent.com/LinneaLager/Prophet_Covid_19/main/streamlit.csv')

	return covid_data



with header:
	st.title('Welcome to my awesome data science project!')
	st.text('In this project I look into the transactions of taxis in NYC. ...')


with dataset:
	st.header('NYC taxi dataset')
	st.text('I foudn this dataset on blablabla.com, ...')

	covid_data = get_data('https://raw.githubusercontent.com/LinneaLager/Prophet_Covid_19/main/streamlit.csv')
	st.write(covid_data.head())

	st.subheader('Pick-up location ID distribution on the NYC dataset')
	pulocation_dist = pd.DataFrame(covid_data['y'].value_counts()).head(50)
	st.bar_chart(pulocation_dist)


with features:
	st.header('The features I created')

	st.markdown('* **first feature:** I created this feature because of this... I calculated it using this logic..')
	st.markdown('* **second feature:** I created this feature because of this... I calculated it using this logic..')



with model_training:
	st.header('Time to train the model!')
	st.text('Here you get to choose the hyperparameters of the model and see how the performance changes!')

	sel_col, disp_col = st.beta_columns(2)

	max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=0, max_value=60000, value=20, step=10)

	n_estimators = sel_col.selectbox('How many trees should there be?', options=['No limit'], index = 0)


	sel_col.text('Here is a list of features in my data:')
	sel_col.write(covid_data.columns)

	input_feature = sel_col.text_input('Which feature should be used as the input feature?','PULocationID')


	if n_estimators == 'No limit':
		regr = RandomForestRegressor(max_depth=max_depth)
	else:
		regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)


	X = covid_data[['ds']]
	y = covid_data[['y']]

	regr.fit(X, y)
	prediction = regr.predict(y)

	disp_col.subheader('Mean absolute error of the model is:')
	disp_col.write(mean_absolute_error(y, prediction))

	disp_col.subheader('Mean squared error of the model is:')
	disp_col.write(mean_squared_error(y, prediction))

	disp_col.subheader('R squared score of the model is:')
	disp_col.write(r2_score(y, prediction))
