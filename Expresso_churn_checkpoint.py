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
from pandas.plotting import scatter_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Expresso_churn_dataset.csv')
st.header("Expresso Churn Prediction")
st.write("This app predict the clients' churn probability.")

data0 = data.copy()

if st.checkbox("Do you wanna display first 4 lines of data"):
	# display the text if the checkbox returns True value
	st.write(data.head())
if st.checkbox("Do you wanna display some statistical information about data"):
    st.write(data.describe())

data.drop('ZONE1', axis=1, inplace=True)
data.drop('DATA_VOLUME', axis=1, inplace=True)
data.drop('ORANGE', axis=1, inplace=True)
data.drop('TIGO', axis=1, inplace=True)
data.drop('user_id', axis=1, inplace=True)
data.drop('MRG', axis=1, inplace=True)
data.drop_duplicates(inplace=True)

label_encoder = preprocessing.LabelEncoder() # import of Label Encoder

data['REGION']= label_encoder.fit_transform(data['REGION']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_REGION = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['TENURE']= label_encoder.fit_transform(data['TENURE']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_TENURE = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

data['TOP_PACK']= label_encoder.fit_transform(data['TOP_PACK']) # transform REGION from categoric to numeric using Label Encoder
mapping_dict_TOP_PACK = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


# Compute correlation

numeric_data = data.select_dtypes(include=["float64"])

if not numeric_data.empty:
    corr = numeric_data.corr()

# Streamlit app
    st.title("Correlation Heatmap")

# Add a description
    st.write("Here is a visulization of the correlation matrix of the provided dataset using a heatmap.")

# Create the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
    corr, 
    annot=True, 
    fmt=".2f",  # Format the annotation text
    cmap="coolwarm",  # Use a diverging colormap
    linewidths=0.5,  # Add space between cells
    cbar_kws={"shrink": 0.8},  # Adjust the colorbar size
    square=True,  # Make cells square
    ax=ax  # Use the ax object
    )

# Customize labels and title
    ax.set_title("Correlation Heatmap", fontsize=16)
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)

# Display the plot in Streamlit
    st.pyplot(fig)

else:
    st.warning("No numeric columns found in the dataset.")


st.write("Please select the Features below.")

REGION = st.selectbox("REGION: ", list(data0['REGION'].unique()))
REGION_map = mapping_dict_REGION[REGION]

TENURE = st.selectbox("TENURE: ", list(data0['TENURE'].unique()))
TENURE_map = mapping_dict_TENURE[TENURE]
 
MONTANT = st.number_input("Enter MONTANT")

FREQUENCE_RECH = st.number_input("Enter FREQUENCE_RECH")

REVENUE = st.number_input("Enter REVENUE")

FREQUENCE = st.number_input("Enter FREQUENCE")

ON_NET = st.number_input("Enter ON_NET")

REGULARITY = st.number_input("Enter REGULARITY")

TOP_PACK = st.selectbox("TOP_PACK: ", list(data0['TOP_PACK'].unique()))
TOP_PACK_map = mapping_dict_TOP_PACK[TOP_PACK]

x = data.drop('CHURN',axis=1)
y = data['CHURN']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42 , stratify = y)
tree = tree.DecisionTreeClassifier() # Creation of model
tree.fit(x_train, y_train)

y_pred_train=tree.predict(x_train)
y_pred=tree.predict(x_test)



if st.button("Make Prediction"):

    input_data = {
        'REGION': [REGION_map],
        'TENURE': [TENURE_map],
        'MONTANT': [MONTANT],
        'FREQUENCE_RECH': [FREQUENCE_RECH],
        'REVENUE': [REVENUE],
        'FREQUENCE': [FREQUENCE],
        'ON_NET': [ON_NET],
        'REGULARITY': [REGULARITY],
        'TOP_PACK': [TOP_PACK_map],
    }
    
    
    input_df = pd.DataFrame(input_data, columns=x_train.columns)

    prediction = tree.predict(input_df)[0]
    prediction_proba = tree.predict_proba(input_df)[0]
    
    # Display results
    st.write("### Prediction Result:")
    if prediction == 1:
        st.success("The client is **likely to churn.**")
    else:
        st.error("The client is **unlikely to churn.**")
    
    st.write("### Prediction Probabilities:")
    st.write(f"- Probability of not churning: {prediction_proba[0]:.2f}")
    st.write(f"- Probability of churning: {prediction_proba[1]:.2f}")



