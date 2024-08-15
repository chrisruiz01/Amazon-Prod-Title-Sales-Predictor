# Product Sales Performance Predictor

## Project Overview
This project develops a deep learning model to predict the sales performance of products based on their titles. Using the 'title' column of our dataset, we construct a Convolutional Neural Network (CNN) with Keras, employing binary classification to predict the likelihood of a product selling over 100 units, referred to as 'target'. Our model leverages GloVe word embeddings to enhance the representation of text data.

## Features
- Data preprocessing and feature extraction from product titles
- Text data handling with Keras Tokenizer and sequence padding
- Integration of GloVe pre-trained word embeddings
- CNN architecture with Keras for binary classification
- Model evaluation with precision and recall metrics

## Installation
To set up the project environment, follow these steps:
1. Clone the repository to your local machine.
2. Ensure that you have Python 3.x installed.
3. Install the required dependencies:

pip install -r requirements.txt

4. Download the GloVe embeddings and place them in the appropriate directory as specified in the notebook.

## Usage
The main notebook '3.0-model-building-convolutional-neural-network.ipynb' contains all the code to preprocess the data, build, and evaluate the model. Run each cell in the Jupyter Notebook to reproduce the model training and evaluation process.

## Model Architecture
The model employs a series of convolutional layers followed by max pooling and dense layers to capture the essence of text data. We use a residual-like architecture to enhance the learning of text features. The model utilizes pre-trained GloVe embeddings to capitalize on existing knowledge about word semantics.

## Evaluation
The model is evaluated based on precision and recall metrics. Precision informs us about the accuracy of positive predictions, while recall tells us about the model's ability to find all the positive samples.

## Contributing
Contributions to this project are welcome. Please ensure to update tests as appropriate and adhere to the existing coding style.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any queries or discussions regarding the project, please contact Chris Ruiz at chrisruiz01@gmail.com.# Product-Sales-Performance-Predictor
