# Dynamic vocabulary for RNN experiment
To run the experiment on the [All the news](https://www.kaggle.com/snapcrack/all-the-news) dataset make sure:
* you have CUDA device available (otherwise it will take very long)
* you have 1.5 Gb of disk space to download, unzip and concatenate the dataset
* python >= 3.6 (at least this is the version I used)


```
git clone https://github.com/grinya007/streaming_rnn.git
cd streaming_rnn
pip3 install -r requirements.txt
kaggle datasets download -d snapcrack/all-the-news  # or curl/wget the zip archive
unzip all-the-news.zip
python3
>>> import pandas as pd
>>> df1 = pd.read_csv('articles1.csv')
>>> df2 = pd.read_csv('articles2.csv')
>>> df3 = pd.read_csv('articles3.csv')
>>> df = pd.concat([df1, df2, df3])
>>> df = df.sort_values('date')
>>> df.to_csv('articles_sorted_by_date.csv')
>>> Ctrl+D
./train.py articles_sorted_by_date.csv
```

This will produce some output and generate two .png files with charts of Model perplexity and Unknown word prediction ratio.
