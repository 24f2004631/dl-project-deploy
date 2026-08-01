from config import MODEL_PATH, VOCAB_PATH
import streamlit as st
from classes import Query, Model

# Runtime
model = Model(MODEL_PATH, VOCAB_PATH)

st.title("Scratch MCQ Model", text_alignment="left")
st.caption("Created by Mohit Anand | Roll: 24f2004631")
st.text("The Scratch MCQ model is as name suggests build from scratch using _RNN_ like _LSTM_ at its core to represent the prompts and its subsequent options as semantic embedding representation which will be further fed into a feed-forward network to get our desired result.")

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
    cont = st.container(border=True)
    cont.subheader(query.prompt)
    ans = model.predict(query)
    for opt in ans[:query.prediction_count]:
        if query.option_text:
            cont.text(f"{opt}: {query.options[opt]}")
        else:
            cont.text(opt)

create = st.button("Create a Request...", type="primary", width="stretch", shortcut="Ctrl+Alt+N")
if create:
    get_input()


st.divider()
st.subheader("Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Map@3", "0.96", help="Map@3 score calculated on validation dataset.")
col2.metric("Accuracy", "0.95", help="Accuracy score calculated on validation dataset.") 
col3.metric("F1 Macro", "0.94", help="F1 Macro score calculated on validation dataset.") 
st.divider()

