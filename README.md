# Analyzing enron email data

Using the categorized emails from Enron email dataset, I have tried multiple modeling approaches like Naive Bayes, linear Support Vector Machine, Random forest classifier and logistic regression approach to classify the emails into the respective categories. The data can be downloaded [here](http://bailando.sims.berkeley.edu/enron/enron_with_categories.tar.gz).

## Steps involved in processing:

- Reading the email content files and respective categories
- Data Cleanup and feature extraction
- Creating TFIDF features (The TfidfVectorizer produces a numpy ndarray with a dimension of (1702, 20933) where 1702 is the number of documents and 20933 is the count of unique words across all the documents and each entry corresponds to a TF-IDF score of a word document pair
- Run the models on the TFIDF features and then perfrom grid search to tune hyperparamenters
- Predictions  

`helper.py` contains all the user defined functions used in modeling.  
`enron.ipynb` performs data cleaning and feature extraction followed by building TFIDF and then uses the above mention modeling techniques for classification. The details regarding the hyperparameter tuning and confusion matrices are present in the notebook.   

## Conclusion:
Accuracy from all four methods:   
Linear SVM : 64.94%   
Logistic regression : 67.90%    
Naive Bayes : 66.97%    
Random Forest : 61.25%     
I got the highest accuracy ( around 68%) from logistic regression.
