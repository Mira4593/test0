import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression 


data = pd.read_csv('Financial_inclusion_dataset.csv')
st.header("Financial Inclusion In Africa")
st.write("This app predicts whether an individual is likely to have a bank account based on demographic and financial data.")

data0 = data.copy()


if st.checkbox("Do you wanna display first 4 lines of data"):
	# display the text if the checkbox returns True value
	st.write(data.head())
if st.checkbox("Do you wanna display some statistical information about data"):
    st.write(data.describe())

label_encoder = preprocessing.LabelEncoder() # import of Label Encoder

data['bank_account']= label_encoder.fit_transform(data['bank_account']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_bank_account = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['country']= label_encoder.fit_transform(data['country']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_country = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['location_type']= label_encoder.fit_transform(data['location_type']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_location_type = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['cellphone_access']= label_encoder.fit_transform(data['cellphone_access']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_cellphone_access = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['gender_of_respondent']= label_encoder.fit_transform(data['gender_of_respondent']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_gender_of_respondent = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['relationship_with_head']= label_encoder.fit_transform(data['relationship_with_head']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_relationship_with_head = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['marital_status']= label_encoder.fit_transform(data['marital_status']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_marital_status = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['education_level']= label_encoder.fit_transform(data['education_level']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_education_level = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['job_type']= label_encoder.fit_transform(data['job_type']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_job_type = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


st.write("Please select the Features below.")

country = st.selectbox("Country: ", list(data0['country'].unique()))
country_map = mapping_dict_country[country]

location_type = st.selectbox("Location_type: ",
						list(data0['location_type'].unique()))
location_type_map = mapping_dict_location_type[location_type]

cellphone_access = st.selectbox("Cellphone_access: ",
						list(data0['cellphone_access'].unique()))
cellphone_access_map = mapping_dict_cellphone_access[cellphone_access]
gender_of_respondent = st.selectbox("Select Gender: ", 
						list(data0['gender_of_respondent'].unique()))
gender_of_respondent_map = mapping_dict_gender_of_respondent[gender_of_respondent]

Age = st.number_input("Enter age")

household_size = st.number_input("household_size")
    
relationship_with_head = st.selectbox("Relationship_with_head: ",
						list(data0['relationship_with_head'].unique()))
relationship_with_head_map = mapping_dict_relationship_with_head[relationship_with_head]

marital_status = st.selectbox("marital_status: ",
						list(data0['marital_status'].unique()))
marital_status_map = mapping_dict_marital_status[marital_status]

education_level = st.selectbox("education_level: ",
						list(data0['education_level'].unique()))
education_level_map = mapping_dict_education_level[education_level]

job_type = st.selectbox("job_type: ",
						list(data0['job_type'].unique()))
job_type_map = mapping_dict_job_type[job_type]

year = st.selectbox("year: ",
					list(data['year'].unique()))


X = data.drop('bank_account', axis=1)  
y = data['bank_account']

X = X.select_dtypes(include=[np.number])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

if st.button("Make Prediction"):

    input_data = {
        'country': [country_map],
        'location_type': [location_type_map],
        'cellphone_access': [cellphone_access_map],
        'gender_of_respondent': [gender_of_respondent_map],
        'age_of_respondent': [Age],
        'household_size': [household_size],
        'relationship_with_head': [relationship_with_head_map],
        'marital_status': [marital_status_map],
        'education_level': [education_level_map],
        'job_type': [job_type_map],
        'year': [year],
    }
    
    
    input_df = pd.DataFrame(input_data, columns=X_train.columns)
    
    
    prediction = clf.predict(input_df)[0]
    prediction_proba = clf.predict_proba(input_df)[0]
    
    # Display results
    st.write("### Prediction Result:")
    if prediction == 1:
        st.success("The individual is **likely to have a bank account.**")
    else:
        st.error("The individual is **unlikely to have a bank account.**")
    
    st.write("### Prediction Probabilities:")
    st.write(f"- Probability of not having a bank account: {prediction_proba[0]:.2f}")
    st.write(f"- Probability of having a bank account: {prediction_proba[1]:.2f}")
