import streamlit as st
from langchain.llms import OpenAI
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

st.title('🦜🔗 Quickstart App')


def generate_response(input_text):
    model_path = 'gaussalgo/T5-LM-Large-text2sql-spider'

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model_inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**model_inputs, max_length=512)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.info(llm(input_text))
    ]
    with st.form('my_form'):
    text = st.text_area('Enter text:', 'What are the three key pieces of advice for learning how to code?')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        generate_response(text)