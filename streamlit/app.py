import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
import streamlit as st

model = load_model('model_objects/model.h5')

# Load the vocabulary
with open("model_objects/vectorizer_vocabulary.pickle", 'rb') as file:
    loaded_vocab = pickle.load(file)

# Define the TextVectorization layer with the loaded vocabulary
max_features = 10000  # Ensure this matches what you used during training
max_text_length = 32  # Ensure this matches what you used during training

vectorizer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=max_text_length,
    vocabulary=loaded_vocab
)

# Function to preprocess text input
def preprocess(text):
    sequences = vectorizer([text])
    return sequences.numpy()
    # sequences = tokenizer.texts_to_sequences([text])  # Wrap text in a list
    # padded_sequences = pad_sequences(sequences, maxlen=max_len)
    # return padded_sequences

# Function to get predictions from the model
def get_predictions(text):
    X = preprocess(text)
    y = model.predict(X)
    predictions = y.tolist()
    return predictions

# Main Streamlit app
def main():
    st.title('Product Sales Performance Predictor')

    st.markdown("""
    **How to Use:**
    1. **Enter Product Title**: Type the title of your Amazon product in the text box. Be as accurate as possible—include brand, model, and key descriptors.
    2. **Get Predictions**: Click the 'Predict Sales Performance' button to estimate the sales potential of your product.
    """)

    input_text = st.text_input(label='Product Title')

    if st.button('Predict Sales Performance'):
        if input_text:
            predictions = get_predictions(input_text)

            # Assume predictions is a list of lists with one item
            if isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], list) and len(predictions[0]) == 1:
                prediction_value = predictions[0][0]
                prediction_percentage = round(prediction_value * 100)
                message = f"Based on the provided product title, there is a {prediction_percentage}% probability that the product titled \"{input_text}\" will sell more than 100 units. Use this insight to optimize your product titles and improve sales potential."
                st.success(message)
            else:
                st.error("The prediction format is not recognized.")
        else:
            st.warning("Please enter a product title to get a prediction.")

    st.markdown("""
    **Tips for Optimization**:
    - **Reflect Key Features**: Include significant features that a potential customer might search for.
    - **Brand and Model**: Always include the brand and model name for better recognition.
    - **Descriptive Keywords**: Use keywords that accurately describe the product's unique qualities and uses.

    Remember, the title is often a customer's first impression of your product—make it count!
    """)

if __name__ == "__main__":
    main()