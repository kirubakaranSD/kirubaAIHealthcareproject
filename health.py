import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # required for BERT tokenizer
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForQuestionAnswering
from tensorflow.keras import layers

# Load the pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = TFBertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Example healthcare knowledge base (medical FAQ)
healthcare_faq = {
    "What are the symptoms of flu?": "Flu symptoms include fever, cough, sore throat, runny or stuffy nose, body aches, headache, chills, and fatigue.",
    "What is the treatment for a cold?": "The treatment for a cold includes rest, hydration, over-the-counter medications to relieve symptoms like pain and fever.",
    "What is diabetes?": "Diabetes is a disease that occurs when blood sugar (glucose) levels are too high. The body doesn't produce enough insulin or doesn't use it properly.",
    "How can I lose weight?": "To lose weight, you can eat a balanced diet, exercise regularly, and maintain a calorie deficit (burning more calories than you consume)."
}

# Convert the FAQ into a pandas dataframe for easy querying
faq_df = pd.DataFrame(list(healthcare_faq.items()), columns=["Question", "Answer"])

# Define a function for question answering
def get_answer_from_faq(question):
    # Find the best match for the question from the FAQ (simple search)
    best_match = faq_df.iloc[(faq_df["Question"].str.contains(question, case=False)).argmax()]
    return best_match['Answer']

# Define a TF-UDF (User Defined Function) for healthcare question answering
@tf.function
def tf_qa_function(input_question):
    input_ids = tokenizer.encode(input_question, return_tensors='tf')
    output = model(input_ids)
    start_positions = tf.argmax(output.start_logits, axis=-1)
    end_positions = tf.argmax(output.end_logits, axis=-1)
    answer_tokens = input_ids[0][start_positions:end_positions+1]
    answer = tokenizer.decode(answer_tokens)
    return answer

# Function to get the answer from the healthcare assistant
def healthcare_assistant(question):
    # First, check the FAQ for simple questions
    answer = get_answer_from_faq(question)
    if answer != "Not found":
        return answer
    # If not in FAQ, use the BERT-based model to provide a detailed answer
    else:
        return tf_qa_function(question).numpy().decode('utf-8')

# Main function to run the assistant
if __name__ == "__main__":
    print("Welcome to the Healthcare Assistant!")
    while True:
        user_input = input("Ask a healthcare question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        else:
            response = healthcare_assistant(user_input)
            print(f"Answer: {response}")
