import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4

def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

def load_data(filename):
    """
    Load shopping data from a CSV file and convert into numeric values.
    """
    months = {
        "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5,
        "Jul": 6, "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
    }

    evidence = []
    labels = []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Evidence: converting columns to correct numeric types
            evidence.append([
                int(row["Administrative"]),
                float(row["Administrative_Duration"]),
                int(row["Informational"]),
                float(row["Informational_Duration"]),
                int(row["ProductRelated"]),
                float(row["ProductRelated_Duration"]),
                float(row["BounceRates"]),
                float(row["ExitRates"]),
                float(row["PageValues"]),
                float(row["SpecialDay"]),
                months[row["Month"]],
                int(row["OperatingSystems"]),
                int(row["Browser"]),
                int(row["Region"]),
                int(row["TrafficType"]),
                1 if row["VisitorType"] == "Returning_Visitor" else 0,
                1 if row["Weekend"] == "TRUE" else 0
            ])
            # Label: 1 if Revenue is true, 0 otherwise
            labels.append(1 if row["Revenue"] == "TRUE" else 0)

    return (evidence, labels)

def train_model(evidence, labels):
    """
    Returns a fitted k-nearest neighbor model (k=1).
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model

def evaluate(labels, predictions):
    """
    Returns (sensitivity, specificity).
    """
    true_positives = 0
    false_negatives = 0
    true_negatives = 0
    false_positives = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1: # Actual purchase
            if predicted == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else: # Actual no purchase
            if predicted == 0:
                true_negatives += 1
            else:
                false_positives += 1

    # Sensitivity: proportion of actual positives correctly identified
    sensitivity = true_positives / (true_positives + false_negatives)
    
    # Specificity: proportion of actual negatives correctly identified
    specificity = true_negatives / (true_negatives + false_positives)

    return (sensitivity, specificity)

if __name__ == "__main__":
    main()