# Import necessary libraries
import datetime  # Often used for timestamping logs or model names (though not explicitly used here later)
import os        # Provides functions for interacting with the operating system (like path joining, checking existence)
import shutil    # High-level file operations (like removing directories)

import pandas as pd  # Library for data manipulation and analysis, particularly DataFrames
import tensorflow as tf # The core TensorFlow library for numerical computation and machine learning
from tf_keras.callbacks import TensorBoard, EarlyStopping # Callbacks for monitoring training (TensorBoard for logs, EarlyStopping could be added)
from tensorflow_hub import KerasLayer # Special Keras layer to load models from TensorFlow Hub
from tf_keras.layers import Dense, Input  # Standard fully connected layer in Keras
from tf_keras.models import Sequential # Linear stack of layers model type in Keras
from tf_keras.preprocessing.text import Tokenizer # Utility for tokenizing text (though not used here as TF Hub handles it)
from tf_keras.utils import to_categorical # Utility function to convert class vectors to binary class matrices (one-hot encoding)
from tf_keras import layers

# Define constants for directory paths
MODEL_DIR = "./text-models" # Directory to save trained models and logs
DATA_DIR = "./data"         # Directory where the input data resides

# Define dataset details
DATASET_NAME = "titles_full.csv" # Name of the dataset file
TITLE_SAMPLE_PATH = os.path.join(DATA_DIR, DATASET_NAME) # Construct the full path to the dataset
COLUMNS = ['title','source'] # Define column names for the CSV file

# Load the dataset using pandas
# header=None signifies the CSV has no header row
# names=COLUMNS assigns the specified column names
titles_df = pd.read_csv(TITLE_SAMPLE_PATH, header=None, names=COLUMNS)

# Display the first few rows of the DataFrame to inspect the data
titles_df.head()

# Display the distribution of the 'source' column (our target labels)
titles_df.source.value_counts()

# Define a mapping from string labels (sources) to integer indices
CLASSES = {
    'github':0,
    'nytimes':1,
    'techcrunch':2
}

# Calculate the total number of unique classes
N_CLASSES = len(CLASSES)

# Define a function to encode string labels into one-hot vectors
def encode_labels(sources):
    """Converts a list of source strings into one-hot encoded vectors."""
    # Convert source strings to their corresponding integer indices using the CLASSES mapping
    classes = [CLASSES[source] for source in sources]
    # Convert the list of integer indices into a one-hot encoded NumPy array
    # num_classes specifies the total number of classes for the one-hot encoding
    one_hots = to_categorical(classes, num_classes=N_CLASSES)
    return one_hots

# Example: Encode and print the first 4 labels to see the one-hot format
print("Example one-hot encoding:")
print(encode_labels(titles_df.source[:4]))

# Define the split point for training data (95% of the dataset)
N_TRAIN = int(len(titles_df) * 0.95)

# Split the DataFrame into training features (titles) and labels (sources)
titles_train, sources_train = (titles_df.title[:N_TRAIN], titles_df.source[:N_TRAIN])

# Split the DataFrame into validation features (titles) and labels (sources)
titles_valid, sources_valid = (titles_df.title[N_TRAIN:], titles_df.source[N_TRAIN:])

# Print the distribution of classes in the training set
print("\nTraining set class distribution:")
print(sources_train.value_counts())

# Print the distribution of classes in the validation set
print("\nValidation set class distribution:")
print(sources_valid.value_counts())

# Prepare the final training data: Convert titles to NumPy array and encode labels
X_train, Y_train = titles_train.values, encode_labels(sources_train)
# Prepare the final validation data: Convert titles to NumPy array and encode labels
X_valid, Y_valid =  titles_valid.values, encode_labels(sources_valid)

# Print the first 3 encoded training labels to verify
print("\nFirst 3 encoded training labels:")
print(Y_train[:3])


# Define a function to build the Keras model
def build_model(hub_module, name):
    """Builds a Sequential Keras model using a TF Hub layer."""
    # Create a Sequential model (linear stack of layers)
    model =  Sequential([
        hub_module, # The TensorFlow Hub KerasLayer (handles text embedding)
        Dense(16, activation="relu"), # A hidden Dense layer with 16 units and ReLU activation
        Dense(N_CLASSES, activation="softmax") # Output Dense layer with N_CLASSES units and softmax activation for multi-class probabilities
    ], name=name) # Assign a name to the model

    # Compile the model, configuring the training process
    model.compile(
        optimizer='adam', # Adam optimization algorithm
        loss='categorical_crossentropy', # Loss function suitable for one-hot encoded multi-class classification
        metrics=['accuracy'] # Metric to evaluate during training and testing
    )
    return model

# Define a function to train the model and evaluate it
def train_and_evaluate(train_data, val_data, model, batch_size=5000):
    """Trains the model, logs progress with TensorBoard, and returns history."""
    # Unpack the training data tuple
    X_train, Y_train = train_data

    # Set a random seed for TensorFlow operations for reproducibility
    tf.random.set_seed(33)

    # Define the directory to save logs for this specific model run
    model_dir =  os.path.join(MODEL_DIR, model.name)
    # If the directory already exists, remove it to avoid mixing logs from previous runs
    if tf.io.gfile.exists(model_dir):
        print(f"Removing previous logs for {model.name} at {model_dir}")
        tf.io.gfile.rmtree(model_dir) # Use tf.io.gfile for potentially remote filesystem compatibility

    # Train the model using model.fit
    # Note: Original code had a typo 'fir', corrected to 'fit'
    history = model.fit(
        X_train, Y_train,          # Training data (features and labels)
        epochs=50,                 # Number of passes through the entire training dataset
        batch_size=batch_size,     # Number of samples per gradient update
        validation_data = val_data, # Data to evaluate the model on after each epoch
        callbacks=[TensorBoard(model_dir)], # List of callbacks, here only TensorBoard for logging
    )
    # Return the history object which contains training/validation loss and metrics per epoch
    return history

# --- Prepare Data Tuples ---
# Group training and validation data into tuples for easier passing to functions
data = (X_train, Y_train)
val_data = (X_valid, Y_valid)

# --- Train and Evaluate with NNLM ---

# Define the URL for the NNLM (Neural Network Language Model) embedding from TensorFlow Hub
NNLM = "https://tfhub.dev/google/nnlm-en-dim50/2"
# Create a KerasLayer using the NNLM module URL
nnlm_module = KerasLayer(
    NNLM,                   # TF Hub module handle
    output_shape = [50],    # Expected output dimension of the embedding (specified by the module)
    input_shape=[],         # Input shape is scalar (empty list) because it expects individual strings
    dtype=tf.string,        # Expected input data type is string
    trainable=True          # Allow the weights of the NNLM module to be fine-tuned during training
)
# Test the layer: Pass a sample sentence (as a tf.constant) to get its embedding
print("\nTesting NNLM layer output shape:", nnlm_module(tf.constant(["The dog is happy to see people in the street."])).shape)


# Build the model using the NNLM KerasLayer
nnlm_model = build_model(nnlm_module, 'nnlm') # Name the model 'nnlm'
# Train and evaluate the NNLM-based model
nnlm_history = train_and_evaluate(data, val_data, nnlm_model)


# --- Plot NNLM Results ---
# Assign the history object for plotting
history = nnlm_history
# Plot training & validation loss using pandas
print("\nPlotting NNLM training history...")
pd.DataFrame(history.history)[['loss','val_loss']].plot(title="NNLM: Loss")
# Plot training & validation accuracy using pandas
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(title="NNLM: Accuracy")


# --- Train and Evaluate with Swivel ---

# Define the URL for the Swivel (Semantic Vector Indexing with Low-Rank Embeddings) embedding from TensorFlow Hub
SWIVEL = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim-with-oov/1"

# Create a KerasLayer using the Swivel module URL
swivel_module =  KerasLayer(
    SWIVEL,                 # TF Hub module handle
    output_shape=[20],      # Expected output dimension (specified by the module)
    input_shape=[],         # Expects scalar strings
    dtype=tf.string,        # Expects string input
    trainable=True          # Allow fine-tuning
)
# Test the Swivel layer
print("\nTesting Swivel layer output shape:", swivel_module(tf.constant(["The dog is happy to see people in the street."])).shape)

# Build the model using the Swivel KerasLayer
swivel_model = build_model(swivel_module, name= 'swivel') # Name the model 'swivel'
# Train and evaluate the Swivel-based model
swivel_history = train_and_evaluate(data, val_data, swivel_model)

# --- Plot Swivel Results ---
# Assign the history object for plotting
history = swivel_history
# Plot training & validation loss
print("\nPlotting Swivel training history...")
pd.DataFrame(history.history)[['loss','val_loss']].plot(title="Swivel: Loss")
# Plot training & validation accuracy
pd.DataFrame(history.history)[['accuracy','val_accuracy']].plot(title="Swivel: Accuracy")

# Import matplotlib if plots don't show automatically in your environment
try:
    import matplotlib.pyplot as plt
    plt.show() # Display the plots
except ImportError:
    print("\nInstall matplotlib (`pip install matplotlib`) to display plots.")