import pandas as pd
import numpy as np

class CustomNaiveBayes:
    def __init__(self):
        self.class_probs = {}

    def train(self, X_train, y_train, word_list):
        num_class_1, total_samples = sum(y_train), len(y_train)
        self.class_probs = {'c1': np.log(num_class_1 / total_samples), 'c0': np.log(1 - num_class_1 / total_samples)}
        class_1, class_0 = X_train[y_train == 1][word_list], X_train[y_train == 0][word_list]
        self.word_given_spam_prob = np.log((class_1.sum(axis=0) + 1) / (class_1.sum().sum() + len(class_1.columns)))
        self.word_given_non_spam_prob = np.log((class_0.sum(axis=0) + 1) / (class_0.sum().sum() + len(class_0.columns)))

    def predict(self, X_test):
        predictions = [1 if np.dot(sample, self.word_given_spam_prob) + self.class_probs['c1'] > np.dot(sample, self.word_given_non_spam_prob) + self.class_probs['c0'] else 0 for _, sample in X_test.iterrows()]
        return predictions

data = pd.read_csv(r"C:\Users\amand\OneDrive\Desktop\emails.csv")
pred_column_index, word_list = data.columns.get_loc('Prediction'), data.columns[1:data.columns.get_loc('Prediction')]
X_data, y_data = data.iloc[:, 1:pred_column_index], data.iloc[:, pred_column_index]
X_train_data, X_test_data = X_data.iloc[:4500], X_data.iloc[4500:]
y_train_data, y_test_data = y_data.iloc[:4500], y_data.iloc[4500:]

nb_classifier = CustomNaiveBayes()
nb_classifier.train(X_train_data, y_train_data, word_list)
predictions = nb_classifier.predict(X_test_data)

confusion_matrix = pd.crosstab(y_test_data, predictions, rownames=['Actual'], colnames=['Predicted'])
correct_classified_spam, correct_classified_non_spam, total_samples = confusion_matrix[1][1], confusion_matrix[0][0], len(y_test_data)
undecided_samples = confusion_matrix.values.sum() - correct_classified_spam - correct_classified_non_spam

print("Training data:")
print(X_train_data, '\n')

print("Testing data:")
print(X_test_data)

print("\nCorrectly classified spam samples =", correct_classified_spam)
print("Correctly classified non-spam samples =", correct_classified_non_spam)
print("Undecided samples =", undecided_samples)
print("Actual spam samples =", y_test_data.sum())
print("Actual non-spam samples =", total_samples - y_test_data.sum())
print("Accuracy =", (correct_classified_spam + correct_classified_non_spam) / total_samples * 100, "%")
