import joblib
#import preprocess_style
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

st.title("Style detection")

labels = ["Mishnah", "Halacha", "Yerushalmi", "Bavli", "Aggadah", "Tanchuma", "Unknown"]
mode = "normal"
# mode = "morph"

model = joblib.load(f"style_normal_logistic_1_3.pkl")
vec = joblib.load(f"style_normal_logistic_vec_1_3.pkl")


default_text = "וידבר ה' אל משה במדבר סיני, זהו שאמר הכתוב. משפטיך תהום רבה, אמר רבי מאיר משל את הצדיקים בדירתן ואת הרשעים בדירתן, משל את הצדיקים בדירתן. במרעה טוב ארעה אתם ובהרי מרום ישראל יהיה נוהם, ומשל את הרשעים בדירתן. כה אמר ה' אלהים ביום רדתו שאולה האבלתי כסתי עליו את"




label = "Enter a paragraph here"

#inp_text = st.text_input(label, value=default_text)
inp_text = st.text_area(label, value=default_text)


import eli5
ex = eli5.explain_prediction(model, inp_text, vec=vec)
exhtml = eli5.formatters.html.format_as_html(ex)
res = exhtml.replace("eli5-weights", "eli5weights").replace("\n", " ")
st.markdown(res, unsafe_allow_html=True)