"""
Topic modelling with Pyspark and spark. The dataset is Amazon musical instruments reviews on Kaggle.
"""

import sparknlp

spark = sparknlp.start()

# We need to select the relevant columns from the dataset, we can use Pyspark for this.

from pyspark.sql import function as F

path = './dataset/musical_instruments_reviews.csv'
data = spark.read.csv(path, header=None)

text_col = 'reviewText'
review_text = data.select(text_col).filter(F.col(text_col).isNotNull())

# transform our data to annotations format so sparknlp can understand

from sparknlp.base import DocumentAssembler

document_assembler = DocumentAssembler().setInputCol(text_col).setOutputCol('Document')

# tokenize data

from sparknlp.annotators import Tokenizer

tokenizer = Tokenizer().setInputCols(['Document']).setOutputCol('tokenized')

# normalize data

from sparknlp.annotators import Normalizer

normalizer = Normalizer().setInputCols(['tokenized']).setOutputCol('normalized').setLowercase(True)

# lemmatization

from sparknlp.annotators import LemmatizerModel

lemmatizer = LemmatizerModel.pretrained().setInputCols(['normalizer']).setOutputCol('lemmatized')

# remove stop-words

from nltk.corpus import stopwords

stopwords = stopwords.words('english')

from sparknl.annotators import StopWordsCleaner

stopwords_cleaner = StopWordsCleaner().setInputCols(['lemmatized']).setOutputCol('no_stop_lemmatized').setStopWords('stopwords')

# using pos_tagger

from sparknlp.annotators import PerceptronModel

pos_tagger = PerceptronModel.pretrained('pos-anc').setInputCols(['document','lemmatized']).setOutputCol('pos')

# filtering out non meaningful n-grams using Chunker

from sparknlp.annotators import Chunker

allowed_tags = ['<JJ>+<JN>', '<NN>+<NN>']

chunker = Chunker().setInputCols(['docoment','pos']).setOutputCol('ngrams').setRegexParsers(allowed_tags)

# transform from annotation format to human readable 

from sparknlp.base import Finisher

finisher = Finisher().setInputCols(['unigrams', 'ngrams'])

# create pipeline

frim pyspark.ml import Pipeline

pipeline = Pipeline().setStages([document_assembler,
    tokenizer, 
    normalizer,
    lemmatizer,
    stopwords_cleaner,
    pos_tagger,
    chunker,
    finisher])

# fitting the data

processed_review = pipeline.fit(review_text).transform(review_text)

# combine unigrams and ngrams

from pyspark.sql.functions import concat

processed_review = processed_review.withColumn('final', concat(F.col('finished_unigrams'), F.col('finished_unigrams')))

# vectorizer: convert textual data into numeric one

from pyspark.ml.feature import CountVectorizer

count_vect = CountVectorizer(inputCol='finished_no_stop_lemmatized', outputCol='tf_features')

tf_model = count_vect.fit(processed_review)
tf_result = tf_model.transform(processed_review)

# inverse frequency of documents

from pyspark.ml.feature import IDF

idf = IDF(inputCol='tf_features', outputCol='tf_idf_features')

tfidf_model = idf.fit(tf_result)
tfidf_result = tfidf_model.transform(tf_result)

# topic modelling with Pyspark using LDA algorithm

from pyspark.ml.clustering import LDA

num_topics = 6
max_iter = 10

lda = LDA(k=num_topics,
          maxIter=max_iter,
          featuresCol='tfidf_features')

lda_model = lda.fit(tfidf_result)

# convert word ids into words

vocab = tf_model.vocabulary

def get_words(token_list):
    return [vocab[token_id] for token_id in token_list]

udf_to_words =  F.udf(get_words, T.ArrayType(T.StringType()))

# show 7 most relevant words to a topic

num_words = 7

topics = lda_model.describeTopics(num_words).withColumn('topicWords',udf_to_words=(F.col('termIndices')))

topics.select('topic', 'topicwords').show(truncate=100)







