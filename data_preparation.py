import os
import db_dtypes

from google.cloud import bigquery
import pandas as pd
from pandas import DataFrame

PROJECT =  "ml-apps-sinni12"
BUCKET = "nlp-reusable-embeddings"
REGION = "us-central1"

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] =  REGION

# query = "SELECT url, title, score FROM  `bigquery-public-data.hacker_news.full` WHERE LENGTH(title) > 10 AND score > 10 AND LENGTH(url) >0 LIMIT 500"
# bg_client = bigquery.Client(project=PROJECT)
# result:DataFrame = bg_client.query(query).to_dataframe()
# result.to_csv("data/hacker-news.csv")
#print(result.head(5))

# df = pd.read_csv("data/hacker-news.csv")
# print(df.head(5))

regex = '.*://(.[^/]+)/'

subquery = """
SELECT 
    title, ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '{0}'),'.'))[safe_offset(1)] AS source
FROM 
    `bigquery-public-data.hacker_news.full`
WHERE 
    REGEXP_CONTAINS(REGEXP_EXTRACT(url, '{0}'), '.com$')
""".format(regex)

query = """
SELECT 
    LOWER(REGEXP_REPLACE(title, '[^a-zA-Z0-9 $.-]', ' ')) AS title,
    source
FROM 
    ({subquery})
WHERE (source = 'github' OR source = 'nytimes' OR source = 'techcrunch')
""".format(subquery=subquery)

print(query)

title_dataset = bigquery.Client(project=PROJECT).query(query).to_dataframe()
print("The full dataset contains {n} titles".format(n=len(title_dataset)))
print(title_dataset.source.value_counts())

DATADIR = './data/'

if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

FULL_DATASET_NAME = 'titles_full.csv'
FULL_DATASET_PATH=  os.path.join(DATADIR, FULL_DATASET_NAME)

title_dataset = title_dataset.sample(n=len(title_dataset))
title_dataset.to_csv(FULL_DATASET_PATH, header= False, index = False, encoding= 'utf-8')

sample_title_dataset: DataFrame = title_dataset.sample(n=1000)
print(sample_title_dataset.source.value_counts())

SAMPLE_DATASET_NAME = 'titles_sample.csv'
SAMPLE_DATASET_PATH = os.path.join(DATADIR, SAMPLE_DATASET_NAME)

sample_title_dataset.to_csv(SAMPLE_DATASET_PATH, header=False, index=False, encoding='utf-8')