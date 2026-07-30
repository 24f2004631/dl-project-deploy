from config import MODEL_PATH, VOCAB_PATH
import streamlit as st
from classes import Query, Model

# Runtime
model = Model(MODEL_PATH, VOCAB_PATH)

st.title("Smart MCQ Predictor")

@st.dialog("Create a request...", width='large')
def get_input():
    st.subheader("Create a request")
    prompt = st.text_input("Question")
    options = {
            "A": "", 
            "B": "", 
            "C": "", 
            "D": "", 
            "E": ""
            }
    for opt in options:
        options[opt] = st.text_input(f"Option {opt}") 

    st.subheader("Additional options...")
    col1, col2 = st.columns([3, 1])
    prediction_count = col1.slider("Number of Prediction", min_value=1, max_value=5, value=3, help="""
                Select the number of options you want to predict.
                """)

    output_text = col2.toggle("Prediction text", help="""
                Toggle if you want option text along with prediction.
                """)
        
    submitted = st.button("Predict", type='primary', icon=':material/arrow_right_alt:', icon_position='right', width='stretch', shortcut='Ctrl+Shift+Enter')
    if submitted:
        st.session_state["query"] = Query(prompt, options, prediction_count, output_text)
        st.rerun()

if "query" in st.session_state:
    query = st.session_state["query"]
    st.header(query.prompt)
    ans = model.predict(query)
    for opt in ans[:query.prediction_count]:
        if query.option_text:
            st.text(f"{opt}: {query.options[opt]}")
        else:
            st.text(opt)

create = st.button("Create a Request...", type="primary")
if create:
    get_input()


