import pandas as pd
import re
# import sklearn.feature_extraction.text as text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

#  Step 1: Load the data
def load_data(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.
    
    Parameters:
    file_path (str): The path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Clean and process the data 
def clean_data(t):
    """
    Clean the perticular data by removing URLs, mentions, Hashtags symbols and unnecessary whitespace.
    
    Parameters:
    t (str): The text to clean.
    
    Returns: text with cleaned data.
    
    """
    t = t.lower()
    t = re.sub(r"http\S+", "", t)              # remove URLs
    t = re.sub(r"@\w+", "", t)                 # remove @mentions
    t = re.sub(r"#", "", t)                    # strip hashtag symbol
    t = re.sub(r"[^a-z0-9\s]", "", t)          # keep letters/numbers/spaces
    t = re.sub(r"\s+", " ", t).strip()         # collapse whitespace
    return t

def preprocess_data(data):
    """
    Preprocess the data by removing unnecessary columns and handling missing values.
    
    Parameters:
    data (pd.DataFrame): The DataFrame to preprocess.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Example preprocessing steps
    data = data.dropna()  # Remove rows with missing values
    data = data.drop_duplicates()  # Remove duplicate rows
    data['cleaned_text'] = data['text'].apply(clean_data)
    # print(data.head(5))
    return data

# Step 3: Vectorize the text data
def vectorize_text(data):
    """
    Vectorize the text data using TF-IDF.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the text data.
    
    Returns:
    scipy.sparse.csr.csr_matrix: The TF-IDF matrix.
    """
    vector = TfidfVectorizer(
        lowercase=False,
        stop_words='english',
        max_features=7000,  # Limit to 5000 features for efficiency
    )
    X = vector.fit_transform(data['cleaned_text'])
    return X, vector

# Step 4: Split the data into training and testing sets
def split_data(data, X):
    """
    Split the data into training and testing sets.
    
    Parameters:
    data (pd.DataFrame): The DataFrame containing the text data and labels.
    
    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    
    return X_train, X_test, y_train, y_test

# Step 5: Train a model
def train_model(X_train, y_train):
    """
    Train a logistic regression model on the training data.
    
    Parameters:
    X_train (scipy.sparse.csr.csr_matrix): The training feature matrix.
    y_train (pd.Series): The training labels.
    
    Returns:
    LogisticRegression: The trained model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

# Step 6: Evaluate the model
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

def predict_sentiment(model, vect, text):
    """
    Predict the sentiment of a given text using the trained model.
    
    Parameters:
    model (LogisticRegression): The trained model.
    text (str): The text to predict sentiment for.
    
    Returns:
    str: The predicted sentiment label.
    """
    cleaned_text = clean_data(text)
    
    X = vect.transform([cleaned_text])
    label = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    prob_dict = dict(zip(model.classes_, probs))

    return label, prob_dict

def example_data(model, vector):
    """
    Example usage of the sentiment analysis workflow.
    """
    print("*** Sentiment Analysis Example Usage ***")
    examples = [
        "I love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "I'm not sure how I feel about this.",
        "Absolutely fantastic service!",
        "Terrible, I will never buy this again."
    ]
    for text in examples:
        cleaned_text = clean_data(text)
        print(f"Original: {text}\nCleaned: {cleaned_text}\n")
        label, probs = predict_sentiment(model, vector, text)
        print(f"Predicted Sentiment: {label}")
        print(f"Probabilities: {probs}\n")


def main():
    """
    Main function to execute the sentiment analysis workflow.
    """
    print("*** Sentiment Analysis Workflow ***")

    # Load and preprocess the data
    print("** Loading data... **")
    file_path = '../data/sentiment_analysis/Tweets.csv'
    data = load_data(file_path)
    
    if data is not None:
        print("Data loaded successfully.")
        print("\n Data shape:", data.shape)
        print("\n Data count:",data.sentiment.value_counts())
        print("\n Sample Data")
        print(data.head(5))
        print("\n** Preprocessing data... **")
        processed_data = preprocess_data(data);
        print("Data preprocessing completed.")
        print("\n Sample Processed Data")
        print(processed_data.head(5))
        print("\n** Vectorizing text data... **")
        X, vector = vectorize_text(processed_data)
        print("Text data vectorization completed.")
        print("\n** Splitting data into training and testing sets... **")
        X_train, X_test, y_train, y_test = split_data(processed_data, X)
        print("Data split completed.")
        print("\n** Training model... **")
        model = train_model(X_train, y_train)
        print("Model training completed.")
        print("\n** Evaluating model... **")
        evaluate_model(model, X_test, y_test)
        print("Model evaluation completed.")
        print("\n*** Sentiment Analysis Workflow Completed ***")

        # Example usage
        example_data(model, vector=vector)
    else:
        print("Failed to load data.")

if __name__ == "__main__":
    main()