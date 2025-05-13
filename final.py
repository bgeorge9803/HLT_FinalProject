# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import time
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

def load_dataset(dataset_type):
    """Load and prepare dataset based on type (email or sms)"""
    if dataset_type == 'email':
        print("Loading data from emails.csv...")
        df = pd.read_csv('emails.csv')
        df = df[['text', 'spam']]
        df.columns = ['message', 'label']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1}).fillna(df['label']).astype(int)
    else:  # SMS
        print("Loading data from sms_dataset.csv...")
        df = pd.read_csv('sms_dataset.csv')
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Print dataset statistics
    total_messages = len(df)
    spam_messages = sum(df['label'])
    ham_messages = total_messages - spam_messages
    print(f"Total messages: {total_messages}")
    print(f"Spam messages: {spam_messages} ({spam_messages/total_messages*100:.2f}%)")
    print(f"Ham messages: {ham_messages} ({ham_messages/total_messages*100:.2f}%)")
    
    return df

def preprocess_text(text):
    """Clean and normalize text data"""
    # Handle NaN values
    if pd.isna(text):
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove Twitter handles
    text = re.sub(r'@\w+', '', text)
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def train_and_evaluate_traditional_models(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression and SVM models"""
    # Lists to store model performance metrics
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    inference_times = []
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Logistic Regression
    print("\n--- Logistic Regression Results ---")
    lr_model = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)
    lr_model.fit(X_train_tfidf, y_train)
    
    # Measure inference time for LR
    start_time = time.time()
    lr_predictions = lr_model.predict(X_test_tfidf)
    end_time = time.time()
    lr_inference_time = (end_time - start_time) / len(X_test) * 1000  # ms per message
    
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_report = classification_report(y_test, lr_predictions, 
                                      target_names=['ham', 'spam'], 
                                      output_dict=True)
    
    print(f"Accuracy: {lr_accuracy:.4f}")
    print(classification_report(y_test, lr_predictions, target_names=['ham', 'spam']))
    print(f"Inference time: {lr_inference_time:.4f} ms per message")
    
    # Store metrics
    model_names.append('Logistic Regression')
    accuracies.append(lr_accuracy * 100)
    precisions.append(lr_report['macro avg']['precision'] * 100)
    recalls.append(lr_report['macro avg']['recall'] * 100)
    f1_scores.append(lr_report['macro avg']['f1-score'] * 100)
    inference_times.append(lr_inference_time)
    
    # SVM
    print("\n--- SVM Results ---")
    svm_model = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
    svm_model.fit(X_train_tfidf, y_train)
    
    # Measure inference time for SVM
    start_time = time.time()
    svm_predictions = svm_model.predict(X_test_tfidf)
    end_time = time.time()
    svm_inference_time = (end_time - start_time) / len(X_test) * 1000  # ms per message
    
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    svm_report = classification_report(y_test, svm_predictions, 
                                       target_names=['ham', 'spam'], 
                                       output_dict=True)
    
    print(f"Accuracy: {svm_accuracy:.4f}")
    print(classification_report(y_test, svm_predictions, target_names=['ham', 'spam']))
    print(f"Inference time: {svm_inference_time:.4f} ms per message")
    
    # Store metrics
    model_names.append('SVM')
    accuracies.append(svm_accuracy * 100)
    precisions.append(svm_report['macro avg']['precision'] * 100)
    recalls.append(svm_report['macro avg']['recall'] * 100)
    f1_scores.append(svm_report['macro avg']['f1-score'] * 100)
    inference_times.append(svm_inference_time)
    
    return {
        'model_names': model_names,
        'accuracies': accuracies,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'inference_times': inference_times,
        'lr_model': lr_model,
        'svm_model': svm_model,
        'tfidf_vectorizer': tfidf_vectorizer
    }

def train_and_evaluate_bert(X_train, X_test, y_train, y_test, num_epochs=5):
    """Train and evaluate BERT model"""
    print("\n--- BERT Model Training ---")
    
    # Load BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Prepare data for BERT
    MAX_LENGTH = 128
    
    def encode_text(texts):
        return tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='tf'
        )
    
    # Encode train and test data
    train_encodings = encode_text(X_train)
    test_encodings = encode_text(X_test)
    
    # Convert to TensorFlow datasets
    batch_size = 16
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        tf.cast(y_train.values, tf.int32)
    )).batch(batch_size)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        tf.cast(y_test.values, tf.int32)
    )).batch(batch_size)
    
    # Early stopping callback to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=2,
        restore_best_weights=True
    )
    
    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
    bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
    
    # Train the model
    print(f"Training BERT for up to {num_epochs} epochs...")
    history = bert_model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=test_dataset,
        callbacks=[early_stopping]
    )
    
    # Store BERT accuracies for each epoch
    bert_accuracies = history.history['val_accuracy']
    
    # If early stopping occurred, fill remaining epochs with the last value
    while len(bert_accuracies) < num_epochs:
        bert_accuracies.append(bert_accuracies[-1])
    
    # Measure inference time for BERT
    start_time = time.time()
    bert_eval_results = bert_model.evaluate(test_dataset, verbose=0)
    end_time = time.time()
    bert_inference_time = (end_time - start_time) / len(X_test) * 1000  # ms per message
    
    # Generate predictions for final BERT model
    bert_predictions = []
    true_labels = []
    
    for batch in test_dataset:
        logits = bert_model(batch[0], training=False).logits
        predictions = tf.argmax(logits, axis=-1)
        bert_predictions.extend(predictions.numpy())
        true_labels.extend(batch[1].numpy())
    
    # Print final results
    bert_accuracy = accuracy_score(true_labels, bert_predictions)
    bert_report = classification_report(true_labels, bert_predictions, 
                                      target_names=['ham', 'spam'], 
                                      output_dict=True)
    
    print(f"\nBERT Final Accuracy: {bert_accuracy:.4f}")
    print(classification_report(true_labels, bert_predictions, target_names=['ham', 'spam']))
    print(f"Inference time: {bert_inference_time:.4f} ms per message")
    
    # Save model
    model_save_path = f'./{dataset_type}_bert_model'
    bert_model.save_pretrained(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    return {
        'bert_model': bert_model,
        'bert_accuracies': bert_accuracies,
        'bert_accuracy': bert_accuracy,
        'bert_precision': bert_report['macro avg']['precision'] * 100,
        'bert_recall': bert_report['macro avg']['recall'] * 100,
        'bert_f1': bert_report['macro avg']['f1-score'] * 100,
        'bert_inference_time': bert_inference_time,
        'history': history
    }

def visualize_results(traditional_results, bert_results, dataset_type, epochs_range):
    """Create visualizations of model performance"""
    # Combine all metrics
    model_names = traditional_results['model_names'] + ['BERT']
    accuracies = traditional_results['accuracies'] + [bert_results['bert_accuracy'] * 100]
    precisions = traditional_results['precisions'] + [bert_results['bert_precision']]
    recalls = traditional_results['recalls'] + [bert_results['bert_recall']]
    f1_scores = traditional_results['f1_scores'] + [bert_results['bert_f1']]
    inference_times = traditional_results['inference_times'] + [bert_results['bert_inference_time']]
    
    # Plot accuracy across epochs
    plt.figure(figsize=(12, 6))
    plt.title(f'{dataset_type.capitalize()} Spam Detection - Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Create constant lines for traditional models
    lr_values = [traditional_results['accuracies'][0]] * len(epochs_range)
    svm_values = [traditional_results['accuracies'][1]] * len(epochs_range)
    
    plt.plot(epochs_range, lr_values, 'b-o', label='Logistic Regression', alpha=0.7)
    plt.plot(epochs_range, svm_values, 'g-o', label='SVM', alpha=0.7)
    plt.plot(epochs_range, [acc * 100 for acc in bert_results['bert_accuracies']], 'r-o', label='BERT', alpha=0.7)
    plt.legend()
    plt.savefig(f'{dataset_type}_accuracy_by_epoch.png', dpi=300, bbox_inches='tight')
    
    # Plot metrics comparison
    plt.figure(figsize=(12, 6))
    plt.title(f'{dataset_type.capitalize()} Spam Detection - Model Metrics Comparison')
    plt.ylabel('Score (%)')
    
    # Bar positions
    x = np.arange(len(model_names))
    width = 0.2
    
    # Create bars
    plt.bar(x - width, precisions, width, label='Precision', color='slateblue', alpha=0.7)
    plt.bar(x, recalls, width, label='Recall', color='mediumseagreen', alpha=0.7)
    plt.bar(x + width, f1_scores, width, label='F1-Score', color='goldenrod', alpha=0.7)
    
    # Add labels and legend
    plt.xticks(x, model_names)
    plt.legend()
    plt.savefig(f'{dataset_type}_model_metrics.png', dpi=300, bbox_inches='tight')
    
    # Plot inference time comparison
    plt.figure(figsize=(12, 6))
    plt.title(f'{dataset_type.capitalize()} Spam Detection - Inference Time Comparison')
    plt.ylabel('Time per message (ms)')
    plt.bar(model_names, inference_times, color=['blue', 'green', 'red'])
    plt.yscale('log')  # Log scale because BERT will be much slower
    plt.savefig(f'{dataset_type}_inference_time.png', dpi=300, bbox_inches='tight')
    
    # Create error analysis visualization
    # For this we'll analyze the BERT model's predictions in more detail
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(true_labels, bert_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['ham', 'spam'], 
                yticklabels=['ham', 'spam'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - BERT ({dataset_type.capitalize()})')
    plt.savefig(f'{dataset_type}_confusion_matrix.png', dpi=300, bbox_inches='tight')

# Main execution block
if __name__ == "__main__":
    # Process both datasets
    for dataset_type in ['email', 'sms']:
        print(f"\n{'='*50}")
        print(f"Processing {dataset_type.upper()} dataset")
        print(f"{'='*50}")
        
        # Load and preprocess dataset
        df = load_dataset(dataset_type)
        df['processed_message'] = df['message'].apply(preprocess_text)
        
        # Split the data into training, validation, and test sets (80-10-10)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            df['processed_message'], 
            df['label'], 
            test_size=0.1,  # 10% for test
            random_state=42, 
            stratify=df['label']
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, 
            y_train_val, 
            test_size=0.11,  # 10% of original data (0.11 * 0.9 = 0.1)
            random_state=42, 
            stratify=y_train_val
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train and evaluate traditional models
        traditional_results = train_and_evaluate_traditional_models(X_train, X_test, y_train, y_test)
        
        # Define epochs range
        epochs_range = range(1, 6)
        
        # Train and evaluate BERT model
        bert_results = train_and_evaluate_bert(X_train, X_test, y_train, y_test, num_epochs=5)
        
        # Visualize results
        visualize_results(traditional_results, bert_results, dataset_type, epochs_range)
        
        # Print final summary
        print("\nFinal Results Summary:")
        print(f"Logistic Regression Accuracy: {traditional_results['accuracies'][0]:.2f}%")
        print(f"SVM Accuracy: {traditional_results['accuracies'][1]:.2f}%")
        print(f"BERT Accuracy: {bert_results['bert_accuracy'] * 100:.2f}%")
