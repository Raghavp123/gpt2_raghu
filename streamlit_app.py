import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 model and tokenizer
##https://www.it-jim.com/blog/training-and-fine-tuning-gpt-2-and-gpt-3-models-using-hugging-face-transformers-and-openai-api/
model_gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function for generating text
## https://huggingface.co/docs/transformers/en/model_doc/gpt2
def generate_text(prompt, num_tokens, temp_value):
    # Tokenize and encode the prompt
   ## https://keras.io/api/keras_nlp/models/gpt2/gpt2_tokenizer/
    input_ids = tokenizer_gpt2.encode(prompt, return_tensors='pt')
    # Generate text with the specified temperature and length
    ##https://huggingface.co/docs/transformers/en/model_doc/gpt2
    ## https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2.py
    ##https://dev.to/whatminjacodes/building-a-simple-chatbot-using-gpt-model-part-2-45cn
        generated_outputs = model_gpt2.generate(
        input_ids,
        max_length=num_tokens,
        temperature=temp_value,
        num_return_sequences=1,
        do_sample=True
    )

    # Decode and return the generated text
    return tokenizer_gpt2.decode(generated_outputs[0], skip_special_tokens=True)

# Streamlit App Interface
##https://github.com/streamlit/streamlit/blob/648b30dd1dd1aab085b9fd868316d6b9d5bd9bdf/examples/interactive_widgets.py#L38
st.title("GPT-2 Predicted and Creative texts")
# Text input user's prompt
user_prompt = st.text_input("Enter your prompt:", value="")
# Token length input
tokens_to_generate = st.number_input("Specify number of tokens:", value=25)

# Generate button
if st.button("Generate Text"):
 if user_prompt:
    # Creative
    st.subheader("Creative Output:")
    creative_output = generate_text(user_prompt, tokens_to_generate, temp_value=1.2)
    st.write(creative_output)

    # Predictable
    st.subheader("Predictable Output:")
    predictable_output = generate_text(user_prompt, tokens_to_generate, temp_value=0.1)
    st.write(predictable_output)

