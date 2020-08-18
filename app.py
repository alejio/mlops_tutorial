import pickle
import numpy as np

import streamlit as st

# The second app
st.markdown('## App 2: Salary Predictor For Techies')
model = pickle.load(open('model.pkl', 'rb'))  # get the model

experience = st.number_input('Years of Experience')
test_score = st.number_input('Aptitude Test score')
interview_score = st.number_input('Interview Score')

features = [experience, test_score, interview_score]


int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


if st.button('Predict'):
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'Your Salary per anum is: Ghc {round(prediction[0], 2)}')