# News-Classifier
Classification of News using naive-bayes classifier
The data.csv file contains 22925 headlines, names of author, short descriptions, dates, and link of news sites. There are three classes: Beauty and Style, Travel, and Business. For predicting the category, links and short descriptions and headlines are enough. I used the laplacian method to avoid 0 possibilities in text classification. For preprocessing the text, I used lemmatization, and after that, I applied stemming on all words to improve accuracy.

I trained the model on 80% of data.csv file and I used 20% for validation. 

For the model created without oversampling:

accuracy is 93.022%, 

Travel Class: precision is 90.8507%, recall is 95.393%

Business Class: precision is 90.8507%, recall is 88.7745%

Style & Beauty: precision is 96,7144%, recall is 93.206%


For the model with oversampling:


accuracy is 93.06%, 

Travel Class: precision is 92.42%, recall is 93.87%

Business Class: precision is 88.08%, recall is 91.95%

Style & Beauty: precision is 97.11%, recall is 92.91%
