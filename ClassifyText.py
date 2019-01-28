from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.classification import LogisticRegression
import pyspark.sql.functions as F
import pyspark.sql.types as T

# Use a local context if you are not on a cluster
sc = SparkContext('local')
spark = SparkSession(sc)

# use this for cluster
#sc = pyspark.SparkContext()
#spark = pyspark.sql.SparkSession(sc)

df = spark.read.format("csv").option("inferschema","true").option("header", "true").option("delimiter", "\t").load("trainReviews.tsv")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(df)
wordsData.show(5)

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
tf = hashingTF.transform(wordsData)
tf.show(10)

tf.head().rawFeatures

idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2).fit(tf)
tfidf = idf.transform(tf)

ml = LogisticRegression(featuresCol="features", labelCol='category', regParam=0.01)
mlModel = ml.fit(tfidf.limit(5000))
res_train = mlModel.transform(tfidf)
extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
res_train.withColumn("proba", extract_prob("probability")).select("id", "proba", "prediction").show()

test_df = spark.read.format("csv").option("inferschema","true").option("header", "true").option("delimiter", "\t").load("testReviews.tsv")

tokenizer = Tokenizer(inputCol="text", outputCol="words")
wordsData = tokenizer.transform(test_df)
wordsData.show(5)

test_tf = hashingTF.transform(wordsData)
test_tf.show(10)

test_idf = IDF(inputCol="rawFeatures", outputCol="features", minDocFreq=2).fit(test_tf)
test_tfidf = idf.transform(test_tf)

test_tfidf.show(5)

res_test = mlModel.transform(test_tfidf)

res_test.show(2)

res_test.withColumn("proba", extract_prob("probability")).select("id", "proba", "prediction").show(10)