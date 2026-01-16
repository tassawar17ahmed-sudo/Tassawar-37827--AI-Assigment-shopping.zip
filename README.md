Shopping AI - Predictor

Project Overview:
This is my implementation for the AI Assignment 2. The goal of this project was to build a machine learning model 
that predicts whether an online shopper will actually complete a purchase based on their browsing behavior.

I used a "k-Nearest Neighbor (k-NN)" classifier to handle this classification task. 
The AI looks at 17 different factors—like how much time a user spent on product pages, their bounce rates, 
and even the month they visited—to decide if they are likely to generate revenue.

How it Works:
1.Data Preprocessing: I wrote a `load_data` function that takes the raw `shopping.csv` data and converts it into numeric values. 
This involved turning months into integers (0-11) and 
converting visitor types and weekend status into binary (0 or 1) so the scikit-learn model could process it[cite: 51, 67, 71, 72].
2.Training: The model is trained using the `KNeighborsClassifier` from the scikit-learn library,
 focusing on the single nearest neighbor to make predictions.
3.Evaluation: The performance is measured using "Sensitivity" (how many actual buyers we caught) and
 "Specificity"(how many non-buyers we correctly identified).


My Results
When I ran the model, I achieved a True Negative Rate (Specificity) of about 90.55% and a True Positive Rate (Sensitivity) of approximately 40.43%. 
This shows the model is quite reliable at identifying who isn't going to buy, while still catching a significant portion of actual intent.