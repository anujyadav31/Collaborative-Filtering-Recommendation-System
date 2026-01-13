import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from ipynb.fs.full.CollaborativeFiltering import movie_recommender_run

#Set page configuration
st.set_page_config(layout = "wide", page_title = "Movie Recommendation App", page_icon = ":Cinema:")

#Write code to call movie_recommender_run and display recommendations

#Display movie rating charts here
#Read the dataset to find unique users
column_names = ['User_ID', 'User_Names','Movie_ID','Rating','Timestamp']
movies_df = pd.read_csv('Movie_data.csv', sep = ',', names = column_names)
n_users = movies_df.User_Names.unique()

#Create application's header
st.header("Collaborative Filtering Recommendation System")

#Create a dropdown of UserIDs
User_Name = st.selectbox(
 "Select a user name:",
 (n_users)
)

st.write("This user might be interested in the following movies:")
#Find and display recommendations for selected users
result = movie_recommender_run(User_Name)
st.table(result.Movie_Title)
