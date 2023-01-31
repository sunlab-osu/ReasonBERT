from pyspark import SparkContext, SparkConf

sc = SparkContext.getOrCreate()

from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType, StringType

from pyspark.sql.types import Row
from pyspark.sql import SparkSession
spark = SparkSession(sc)

import json
import os

split = os.environ["SPLIT"]

data_dir = '/fs/scratch/PAA0201/osu8727/hybrid_pretrain'

linked_sentences = spark.createDataFrame(sc.textFile(os.path.join(data_dir, f'sentences_with_link_cleaned_nonempty.json/part-0{split}*'))\
                                    .map(json.loads)\
                                    .map(lambda x: Row(
                                        title=x['title'],\
                                        wikiTitle=x['wikiTitle'],\
                                        wid=x['wid'],\
                                        sec_i=x['sec_i'],\
                                        p_i=x['p_i'],\
                                        s_i=x['s_i'],\
                                        s=x['linked_sentence'][0],\
                                        links=x['linked_sentence'][1],\
                                        entities=x['linked_sentence'][2])))

from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

from sparknlp.pretrained import PretrainedPipeline, NerDLModel, BertEmbeddings, WordEmbeddingsModel
import sparknlp

documentAssembler = DocumentAssembler() \
    .setInputCol("s") \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.load(os.path.join(data_dir, "models/glove_100d_en_2.4.0_2.4_1579690104032")) \
        .setInputCols("sentence", "token") \
        .setOutputCol("embeddings") \


ner = NerDLModel.load(os.path.join(data_dir, "models/onto_100_en_2.4.0_2.4_1579729071672")) \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")


nerConverter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline() \
    .setStages([
        documentAssembler,
        tokenizer,
        embeddings,
        ner,
        nerConverter
    ])

model = pipeline.fit(linked_sentences)
annotated_sentences = model.transform(linked_sentences)

def add_ner_links(original_links, entities, ner_links, s_length):
    valid_ner = {"DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "CARDINAL"}
    mapped_loc = [False for _ in range(s_length)]
    merged_links = []
    final_ner_links = []
    entities = set(entities)
    for link in original_links:
        start = int(link[2])
        end = int(link[3])
        merged_links.append([link[0],link[1],start,end,'hyper','hyper'])
        mapped_loc[start:end] = [True]*(end-start)
    for link in ner_links:
        start = link.begin
        end = link.end+1
        ner_type = link.metadata['entity']
        e_id = f"{ner_type}:{link.result}"
        final_ner_links.append([e_id,link.result,start,end,'ner',link.metadata['entity']])
        if not any(mapped_loc[start:end]) and ner_type in valid_ner:
            merged_links.append([e_id,link.result,start,end,'ner',link.metadata['entity']])
            entities.add(e_id)
    return [merged_links, list(entities), final_ner_links]

annotated_sentences = annotated_sentences\
    .drop('sentence','token','embeddings','ner')\
    .where((F.size(F.col('links'))+F.size(F.col('ner_chunk')))>1)\
    .rdd.map(lambda x:{
        'wid': x.wid,
        'title': x.title,
        'wikiTitle': x.wikiTitle,
        'sec_i': x.sec_i,
        'p_i': x.p_i,
        's_i': x.s_i,
        's': x.s,
        'links': add_ner_links(x.links, x.entities, x.ner_chunk, len(x.s))
    })\
    .filter(lambda x:len(x['links'][1])>1)

annotated_sentences.map(lambda x:json.dumps(x)).saveAsTextFile(os.path.join(data_dir,f'annotated_sentences.json-part-{split}'))