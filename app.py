import joblib
#import preprocess_style
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import re
import eli5


def filter_non_hebrew(text):
    # The Unicode range for Hebrew is U+0590 to U+05FF.
    # The Unicode range for Hebrew diacritical marks is U+0591 to U+05BD, U+05BF, U+05C1, U+05C2, U+05C4, U+05C5, U+05C7.
    # So, the Unicode range for Hebrew letters without diacritical marks is U+05D0 to U+05EA.
    hebrew_text = re.sub(r'[^\u05D0-\u05EA]', '', text)
    return hebrew_text


st.title("Style detection")

labels = ["Mishnah", "Halacha", "Yerushalmi", "Bavli", "Aggadah", "Tanchuma"]#, "Unknown"]
mode = "normal"
# mode = "morph"

model = joblib.load(f"style_normal_logistic_1_3.pkl")
vec = joblib.load(f"style_normal_logistic_vec_1_3.pkl")


default_text = "וידבר ה' אל משה במדבר סיני, זהו שאמר הכתוב. משפטיך תהום רבה, אמר רבי מאיר משל את הצדיקים בדירתן ואת הרשעים בדירתן, משל את הצדיקים בדירתן. במרעה טוב ארעה אתם ובהרי מרום ישראל יהיה נוהם, ומשל את הרשעים בדירתן. כה אמר ה' אלהים ביום רדתו שאולה האבלתי כסתי עליו את"




label = "Enter a paragraph here"

#inp_text = st.text_input(label, value=default_text)
inp_text = st.text_area(label, value=default_text)
filtered_text = filter_non_hebrew(inp_text)


ex = eli5.explain_prediction(model, filtered_text, vec=vec, target_names=labels)
exhtml = eli5.formatters.html.format_as_html(ex)
res = exhtml.replace("eli5-weights", "eli5weights").replace("\n", " ")
st.markdown(res, unsafe_allow_html=True)