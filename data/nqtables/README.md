NQTables is a subset from Natural Questions that we used for table question answering evaluation, where we filter out examples that can be answered using tables from the provided Wiki article. The processed data could be found [here](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/deng_595_buckeyemail_osu_edu/EZEwhbeFITRBtZ0Vj0GCjlsBCPQ4EgMtFOs-vRrxY04S1w?e=TZ2CAw). We also provided the notebook to create it from the official Natural Questions release.

The processed data includes three versions:
- For Text QA, where we just keep the tables as how they are presented in the original Natural Questions corpus. The tables are essentially linearized and can be handled by normal text-based QA models. `onlytable` splits keep only the tables and remove other parts of the original Wiki article.
    - nqtable_textQA.jsonl
    - nqtable_textQA-onlytable.jsonl
- For Table QA, where we parse the table and store it as list of rows. This is used for models designed for tables. We chunk the table during preprocessing if it is too large.
    - nqtable_row_first.jsonl
