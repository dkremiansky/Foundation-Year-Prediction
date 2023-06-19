# Foundation-Year-Prediction

The task I want to solve is predicting the year of establishment for the companies for which this information is provided
missing (36% of all companies in the data). I would like to do this using the rest of the data we have, while
testing various features used by me for the purpose of forecasting.

During the work and testing of the various models (I will detail them below) I was working on a sample from the total data, where the sample is of the size of about 100,000 records. They were taken only from the companies that have the year of establishment.

My baseline was the completion of the establishment year according to the average of the years of establishment of the industry to which this company belongs. 

The features we test (separately or together) are: vector/numerical representation of the text and company size.

I compared several different approaches to the representation of the text, both in terms of the text represented, both in terms of the type of embedding (embedding) and in terms of aggregation.
I checked whether preprocessing (removing common words, punctuation marks, etc.) of the text improves results or, on the contrary, damages the results. In addition, I checked whether there is a link to the text (processed and unprocessed) of the other textual data (name of the company, industry and location) improve the quality of the results. For each of these options I tested different types of representations for the text. For different representations I tested different aggregations: no aggregation,
Aggregation of sum/average/chain etc.

The representations I used are:
1) tf-idf where max-features = 500
2) Word2Vec that we trained when the parameters of the model are: a vector of length 100, a window of size 5 and training rounds = 100
3) A pre-trained representation model from the Word2Vec library of gensim   named Glove when instead of words that do not appear in the trained model we put a vector of zerosin the appropriate length
4) A model (representation and classification) of the library fastText  with parameters: vector dimension of a word is 50, N - gram = 2 , the loss function is hierarchical softmax and the number of training rounds is 10
5) A model (representation and regression) RoBerta  with the parameters: maximum length of 256, batch size also of the training
and also of the validation is 8, learning - rate = 0.1 and number of training rounds = 5
