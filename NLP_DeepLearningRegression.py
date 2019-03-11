
import os, json
import numpy as np
from operator import itemgetter
import pandas as pd
from nltk.corpus import stopwords
from keras import optimizers
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

# News article data directory
path_to_json = "E:\\Thesis\\DataSets\\Webhose Purchase\\3931_5b19ed4943ea186bb6a8967c96b46299\\"

json_sub_directories = {"2018_01_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_02_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_03_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_04_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_05_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_06_5b19ed4943ea186bb6a8967c96b46299",
                        "2018_07_5b19ed4943ea186bb6a8967c96b46299"
                        }
Data = []

# Load articles from json files into environment
def loadJsonData(json_sub_directories, path_to_json):
    for subdir in json_sub_directories:
        targetDir = path_to_json + subdir
        json_files = [pos_json for pos_json in os.listdir(targetDir) if pos_json.endswith(".json")]
        for File in json_files:
            jsonData = json.loads(open(targetDir + "\\" + File, encoding='utf-8').read())
            Data.append(jsonData)
        sortedData = sorted(Data, key=itemgetter('published'))
    return sortedData;


Data = loadJsonData(json_sub_directories, path_to_json)

# 2018-01-04T20:04:20.007+02:00          %Y-%m-%dT%H:%M:%S.%fZ
# extract required data
cnt = 0
newsDateData = list()
for obj in Data:
    newsDateData.append([obj['uuid'],
                         obj['published'].split('T')[0],
                         obj['title'], obj['text']])

# To pandas DataFrame
newsDateData = pd.DataFrame(newsDateData, columns=['uuid', 'published', 'title', 'text'])
# change string to Datetime
newsDateData.published = pd.to_datetime(newsDateData.published)
newsDateData.to_csv("E:\\Thesis\\DataSets\\Webhose Purchase\\3931_5b19ed4943ea186bb6a8967c96b46299\\combinedData.csv",
                    encoding='utf-8')
# Load historical bitcoin price
bitcoinData = pd.read_csv("E:\\Thesis\\DataSets\\BitcoinHistoricalPriceDec2017_Jul2018.csv")
bitcoinData.Date = pd.to_datetime(bitcoinData.Date)
# merge articles with BTC price data
mergedData = pd.merge(newsDateData, bitcoinData,
                      how='inner',
                      right_on="Date", left_on="published", sort=True)
# sort data by article publication date
finalData = mergedData.sort_values('published')
finalData.to_csv("E:\\Thesis\\DataSets\\Webhose Purchase\\3931_5b19ed4943ea186bb6a8967c96b46299\\FinalData.csv", encoding='utf-8',
                 escapechar='\\')

n = finalData.shape[0] # dataset row count
train_size = 0.3 # split value for train and test

# split data into train and test
np.random.seed(0)
train_dataframe = finalData.iloc[int(n * train_size):]
test_dataframe = finalData.iloc[:int(n * train_size)]

# split target into train and test
targetTrain = train_dataframe['PriceChangePercent'].values
targetTest = test_dataframe['PriceChangePercent'].values

MAX_NB_WORDS = 100000
tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '\\', '/',
                   '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

# Load FastText pre-trained word vectors
import codecs
print('loading word embeddings...')
embeddings_index = {}
f = codecs.open('E:\\Thesis\\DataSets\\wiki-news-300d-1M.vec', encoding='utf-8')
for line in tqdm(f):
    values = line.rstrip().rsplit(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('found %s word vectors' % len(embeddings_index))

# Find avg length of articles
train_dataframe['document_len'] = train_dataframe['text'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(train_dataframe['document_len'].mean() +
                       train_dataframe['document_len'].std()).astype(int)

raw_docs_train = train_dataframe['text'].tolist()
raw_docs_test = test_dataframe['text'].tolist()
# num_classes = 2  # only two classes Fall (0) and Rise (1) Price_Change_cat

# Data pre-processing
print("pre-processing train data...")
processed_docs_train = []
for doc in tqdm(raw_docs_train):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_train.append(" ".join(filtered))

processed_docs_test = []
for doc in tqdm(raw_docs_test):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    processed_docs_test.append(" ".join(filtered))
# Tokenization of text data
print("tokenizing input data...")
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
tokenizer.fit_on_texts((processed_docs_test + processed_docs_train))

word_seq_train = tokenizer.texts_to_sequences(processed_docs_train)
word_seq_test = tokenizer.texts_to_sequences(processed_docs_test)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))

# pad sequences
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)

# training params
batch_size = 100
num_epochs = 50

# model parameters
num_filters = 64
embed_dim = 300
weight_decay = 1e-4

# embedding matrix
print('preparing embedding matrix...')
words_not_found = []
nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
embedding_matrix = np.zeros((nb_words, embed_dim))
for word, i in word_index.items():
    if i >= nb_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if (embedding_vector is not None) and len(embedding_vector) > 0:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        words_not_found.append(word)
print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

print("training CNN ...")
model = Sequential()
model.add(
        Embedding(
                nb_words,
                embed_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=False
                )
        )
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1, activation='sigmoid'))
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam)
model.summary()
#model training
hist = model.fit(word_seq_train, targetTrain, batch_size=batch_size,
                 epochs=10, #validation_split=0.1,
                 validation_data=[word_seq_test, targetTest],
                 shuffle=False, verbose=2)
# Model evaluation
eval = model.evaluate(word_seq_test, targetTest, verbose=2)
# Model Prediction
pred = model.predict(word_seq_test, verbose=1)
print("prediction MSE", eval)
import time
timeStr = time.strftime("%Y_%m_%d-%H_%M_%S")
print("Generating Output files for Model with time ", timeStr)
# export training statistics
pd.DataFrame(hist.history).to_json("E:\\Thesis\\DataSets\\Evaluation\\CNN_last_result_epochs_50_Time_"+timeStr+".json")

#create a submission for Prediction Result
submission_df = pd.DataFrame(columns=['targetTestFlag','OriginalFlagValue'])
submission_df['targetTestFlag'] = pred.tolist()
submission_df['OriginalFlagValue'] = test_dataframe['PriceChangePercent'].tolist()
submission_df.to_csv("E:\\Thesis\\DataSets\\ResultSubmissions\\CNN\\cnn_fasttext_regression"+
                     timeStr+".csv", index=False)

# LSTM Model
from keras.layers import LSTM
print("training LSTM ...")
model = Sequential()
model.add(
        Embedding(
                nb_words,
                embed_dim,
                weights=[embedding_matrix],
                input_length=max_seq_len,
                trainable=False
                )
        )
model.add(LSTM(300, return_sequences=True))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mse'])
model.summary()

#model training
hist = model.fit(word_seq_train, targetTrain, batch_size=batch_size, epochs=num_epochs,
                validation_data=[word_seq_test, targetTest], #validation_split=0.1,
                verbose=2, shuffle=False)

eval = model.evaluate(x=word_seq_test, y=targetTest, verbose=2)
print("prediction MSE", eval)
pred = model.predict(word_seq_test, verbose=1)

# export training statistics
pd.DataFrame(hist.history).to_json("E:\\Thesis\\DataSets\\Evaluation\\LSTM_last_result_epochs_50_Time_"+timeStr+".json")

# create a submission for Prediction Result
submission_df = pd.DataFrame(columns=['targetTestFlag','OriginalFlagValue'])
submission_df['targetTestFlag'] = pred.tolist()
submission_df['OriginalFlagValue'] = test_dataframe['PriceChangePercent'].tolist()
submission_df.to_csv("E:\\Thesis\\DataSets\\ResultSubmissions\\CNN\\lstm_fasttext_regression"+timeStr+".csv", index=False)
