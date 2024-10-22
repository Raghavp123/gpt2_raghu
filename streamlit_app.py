import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function for generating text
def generate_text(prompt, num_tokens, temp_value):
    # Tokenize and encode the prompt
    input_ids = tokenizer_gpt2.encode(prompt, return_tensors='pt')
    # Generate text with the specified temperature and length
    generated_outputs = model_gpt2.generate(
        input_ids,
        max_length=num_tokens,
        temperature=temp_value,
        num_return_sequences=1
    )

    # Decode and return the generated text
    return tokenizer_gpt2.decode(generated_outputs[0], skip_special_tokens=True)

# Streamlit App Interface
st.title("GPT-2 Predicted and Creative texts")
# Text input user's prompt
user_prompt = st.text_input("Enter your prompt:", value="Type here")
# Token length input
tokens_to_generate = st.number_input("Specify number of tokens:", value=50)

# Generate button
if st.button("Generate Text"):
    
    # Creative
    st.subheader("Creative Output:")
    creative_output = generate_text(user_prompt, tokens_to_generate, temp_value=2.7)
    st.write(creative_output)

    # Predictable
    st.subheader("Predictable Output:")
    predictable_output = generate_text(user_prompt, tokens_to_generate, temp_value=1.2)
    st.write(predictable_output)

