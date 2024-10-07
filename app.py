import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource 
def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return tokenizer, model

models = {
    "GPT-Neo-1.3B": "EleutherAI/gpt-neo-1.3B",
    "GPT-2": "openai-community/gpt2",
    "GPT-2 Medium": "openai-community/gpt2-medium"
}

st.title("Efficient Article Generator Bot")
st.subheader("Choose a model and input a topic to generate an article.")

model_choice = st.sidebar.selectbox(
    "Select a model:",
    ("GPT-Neo-1.3B", "GPT-2", "GPT-2 Medium")
)

tokenizer, model = load_model(models[model_choice])

prompt = st.text_input("Enter your article topic or prompt:", "")

max_length = st.slider("Select article length (max tokens):", 100, 500, 300)
temperature = st.slider("Select temperature (creativity):", 0.5, 1.5, 0.7)

def generate_article(prompt, model, tokenizer, max_length=300, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(torch.device('cpu'))
    
    with torch.no_grad():
        output = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(output[0], skip_special_tokens=True)


if st.button("Generate Article"):
    if prompt:
        with st.spinner(f"Generating article with {model_choice}..."):
            article = generate_article(prompt, model, tokenizer, max_length=max_length, temperature=temperature)
        st.success("Article Generated!")
        st.write(f"### Article by {model_choice}")
        st.write(article)
    else:
        st.warning("Please enter a prompt to generate an article.")
