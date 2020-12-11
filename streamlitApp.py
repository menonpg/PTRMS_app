
import requests
import streamlit as st
from PIL import Image

import config
import inference
import base64
from io import BytesIO
import pandas as pd

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1') # <--- here
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    val = to_excel(df)
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="PRTMS_predictions.xlsx">Download results</a>' # decode b'abc' => abc


MODELS = {
    "Best of Family Ensemble 01": "StackedEnsemble_BestOfFamily_AutoML_20201201_220644",
    "Best of Family Ensemble 02": "StackedEnsemble_BestOfFamily_AutoML_20201201_224937",
    "Complete Ensemble": "StackedEnsemble_AllModels_AutoML_20201201_220644",
}

# https://discuss.streamlit.io/t/version-0-64-0-deprecation-warning-for-st-file-uploader-decoding/4465
st.set_option("deprecation.showfileUploaderEncoding", False)

# defines an h1 header
st.title("PTRMS Breath Analyzer")

# displays a file uploader widget
image = st.file_uploader("Upload a CSV input file")

# displays the select widget for the styles
style = st.selectbox("Choose a model", [i for i in MODELS.keys()])

# displays a button
if st.button("Analyze Sample"):
    if image is not None and style is not None:
        files = {"file": image.getvalue()}
        # res = requests.post(f"http://localhost:8081/{style}", files=files)

        output  = inference.inference(config.MODELS[style], image) #files.file)
        # name = f"storage/{str(image)}_output.csv"
        # output.to_csv(name)


        # img_path = res.json()
        # fileloaded = pd.read_csv("name")
        st.markdown(get_table_download_link(output), unsafe_allow_html=True)
        st.write(output)

