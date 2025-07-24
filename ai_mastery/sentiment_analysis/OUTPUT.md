python sentimentAnalysis.py    
*** Sentiment Analysis Workflow ***
** Loading data... **
Error loading data: [Errno 2] No such file or directory: '../data/sentiment_analysis/sentiment_data.csv'
Failed to load data.
(venv) (base) pranavrnayakn@Pranavs-MacBook-Air-2 sentiment_analysis % python sentimentAnalysis.py
*** Sentiment Analysis Workflow ***
** Loading data... **
Data loaded successfully.

 Data shape: (27481, 4)

 Data count: sentiment
neutral     11118
positive     8582
negative     7781
Name: count, dtype: int64

 Sample Data
       textID                                               text                        selected_text sentiment
0  cb774db0d1                I`d have responded, if I were going  I`d have responded, if I were going   neutral
1  549e992a42      Sooo SAD I will miss you here in San Diego!!!                             Sooo SAD  negative
2  088c60f138                          my boss is bullying me...                          bullying me  negative
3  9642c003ef                     what interview! leave me alone                       leave me alone  negative
4  358bd9e861   Sons of ****, why couldn`t they put them on t...                        Sons of ****,  negative

** Preprocessing data... **
Data preprocessing completed.

 Sample Processed Data
       textID                                               text  ... sentiment                                       cleaned_text
0  cb774db0d1                I`d have responded, if I were going  ...   neutral                  id have responded if i were going
1  549e992a42      Sooo SAD I will miss you here in San Diego!!!  ...  negative         sooo sad i will miss you here in san diego
2  088c60f138                          my boss is bullying me...  ...  negative                             my boss is bullying me
3  9642c003ef                     what interview! leave me alone  ...  negative                      what interview leave me alone
4  358bd9e861   Sons of ****, why couldn`t they put them on t...  ...  negative  sons of why couldnt they put them on the relea...

[5 rows x 5 columns]

** Vectorizing text data... **
Text data vectorization completed.

** Splitting data into training and testing sets... **
Data split completed.

** Training model... **
Model training completed.

** Evaluating model... **
Confusion Matrix:
[[ 905  571   80]
 [ 280 1703  240]
 [  64  510 1143]]

Classification Report:
              precision    recall  f1-score   support

    negative       0.72      0.58      0.65      1556
     neutral       0.61      0.77      0.68      2223
    positive       0.78      0.67      0.72      1717

    accuracy                           0.68      5496
   macro avg       0.71      0.67      0.68      5496
weighted avg       0.70      0.68      0.68      5496

Model evaluation completed.

*** Sentiment Analysis Workflow Completed ***
*** Sentiment Analysis Example Usage ***
Original: I love this product! It's amazing.
Cleaned: i love this product its amazing

Predicted Sentiment: positive
Probabilities: {'negative': np.float64(0.0010788011393969572), 'neutral': np.float64(0.008265238532137599), 'positive': np.float64(0.9906559603284656)}

Original: This is the worst experience I've ever had.
Cleaned: this is the worst experience ive ever had

Predicted Sentiment: negative
Probabilities: {'negative': np.float64(0.782289779325676), 'neutral': np.float64(0.14893854573736773), 'positive': np.float64(0.06877167493695624)}

Original: I'm not sure how I feel about this.
Cleaned: im not sure how i feel about this

Predicted Sentiment: negative
Probabilities: {'negative': np.float64(0.6498594164942189), 'neutral': np.float64(0.2244580678972195), 'positive': np.float64(0.12568251560856178)}

Original: Absolutely fantastic service!
Cleaned: absolutely fantastic service

Predicted Sentiment: positive
Probabilities: {'negative': np.float64(0.06161502316958769), 'neutral': np.float64(0.14595559653152992), 'positive': np.float64(0.7924293802988824)}

Original: Terrible, I will never buy this again.
Cleaned: terrible i will never buy this again

Predicted Sentiment: negative
Probabilities: {'negative': np.float64(0.7716497834677295), 'neutral': np.float64(0.200077339607246), 'positive': np.float64(0.028272876925024504)}
