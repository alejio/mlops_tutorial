from joblib import load
import numpy as np

import streamlit as st

st.markdown('## Human activity predictor from smartphones')
model = load('models/decision_tree.joblib')  # get the model

feature_a = st.number_input('tBodyAccMag-mean()')
feature_b = st.number_input('angle(X,gravityMean)')
feature_c = st.number_input('angle(Y,gravityMean)')

total_features = 561

int_features = [0 for i in range(total_features)]
int_features[200] = feature_a
int_features[558] = feature_b
int_features[559] = feature_c

final_features = [np.array(int_features)]


if st.button('Predict'):
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'The predicted activity is {prediction[0]}')