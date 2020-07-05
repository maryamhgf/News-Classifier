# News-Classifier
In this project we consider classification of news using the naive-bayes classifier. The data.csv file contains 22925 headlines, names of authors, short descriptions, dates, and a link of news sites. There are three classes: Beauty and Style, Travel, and Business. For predicting the category, links and short descriptions and headlines are considered. I used the Laplacian method to avoid “zero possibilities” in the text classification. For preprocessing the text, I used lemmatization, and after that, I applied stemming on all words to improve accuracy.
I trained the model on 80% of data.csv file and used 20% for validation.
For the model created without oversampling:
accuracy is 93.022%,
Travel Class: precision=90.8507%, recall=95.393%
Business Class: precision=90.8507%, recall=88.7745%
Style & Beauty: precision=96,7144%, recall=93.206%
For the model with oversampling:
accuracy is 93.06%,
Travel Class: precision=92.42%, recall=93.87%
Business Class: precision=88.08%, recall=91.95%
Style & Beauty: precision is 97.11%, recall=92.91%
(data.csv was downloaded from kaggle.com)


