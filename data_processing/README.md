This folder contains the scripts used to prepare the data for pretraining. You can follow the insutructions in `process_article_sentence_only.ipynb` to create the pretraining data for `ReasonBERT-RoBERTa`. Here we summarize the main steps and intermediate files generated in the process. Some of the prepared files are shared [here](https://buckeyemailosu-my.sharepoint.com/:f:/g/personal/deng_595_buckeyemail_osu_edu/EsRBBK6mn59Ju-Vg3juzYScB-ZssR9jTPfpU0qxkeIaBgw?e=kVh1Zn).

1. The initial Wikipedia XML dump is processed using the tool developped `https://github.com/idio/json-wikipedia` and converted into json format.

       enwiki-20201201-pages-articles.json

2. We process the Wikidata dump to find aliases for wikipedia entities, which are later used for entity linking.

        wikidata_aliases.json

3. We extract paragraphs from the pages, add back the hyperlinks and do some basic cleaning.

        paragraphs_with_link_cleaned.json

4. We use `SparkNLP` for sentence boundary detection and add hyperlinks to the sentences.

        sentences_with_link_cleaned_nonempty.json
        {
            'wid': x.wid,
            'title': x.title,
            'wikiTitle': x.wikiTitle,
            'linked_sentence': y[:3],
            'sec_i': x.sec_i, # section index
            'p_i': x.p_i, # paragraph index
            's_i': y[3], # sentence index
            'md5': hashlib.md5((x.wikiTitle+'$$%md5%$$'+y[0]).encode()).hexdigest()
        }

5. We use `SparkNLP` for NER, this steps may take some a while. The `slurm_jobs` contains some scripts we used to run the job on a slurm cluster.

        annotated_sentences.json-part-*
        root
        |-- entities: array (nullable = true)
        |    |-- element: string (containsNull = true)
        |-- links: array (nullable = true)
        |    |-- element: array (containsNull = true)
        |    |    |-- element: string (containsNull = true)
        |-- s: string (nullable = true)
        |-- title: string (nullable = true)
        |-- wid: long (nullable = true)
        |-- wikiTitle: string (nullable = true)
        |-- sentence: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- annotatorType: string (nullable = true)
        |    |    |-- begin: integer (nullable = false)
        |    |    |-- end: integer (nullable = false)
        |    |    |-- result: string (nullable = true)
        |    |    |-- metadata: map (nullable = true)
        |    |    |    |-- key: string
        |    |    |    |-- value: string (valueContainsNull = true)
        |    |    |-- embeddings: array (nullable = true)
        |    |    |    |-- element: float (containsNull = false)
        |-- token: array (nullable = true)
        |    |-- element: struct (containsNull = true)
        |    |    |-- annotatorType: string (nullable = true)
        |    |    |-- begin: integer (nullable = false)
        ...
        |    |    |    |-- value: string (valueContainsNull = true)
        |    |    |-- embeddings: array (nullable = true)
        |    |    |    |-- element: float (containsNull = false)

6. We combine NER entities and original hyperlinks to generate the entity pairs and filter the sentences. We keep only sentences that mention the topic entity of that Wikipedia page.

        annotated_sentences_onlyself_withpairs.json

7. We join the sentences use entity pairs and form the query-evidence pairs as described in the paper.

        annotated_sentences_onlyself_spairs_noskew.json

8. Generate and write the final pretraining data.

        sentence_multi_pairs_for_pretrain_no_tokenization/%06d.tar
