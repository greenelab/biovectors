#!/usr/bin/env python
# coding: utf-8

# # Get Publication Times for Pubmed Abstracts

# In[1]:


import csv
from pathlib import Path
import time

import pandas as pd
import requests
import tqdm


# In[2]:


# Write the api caller function
def call_entrez(pubmed_ids):
    while True:
        try:
            url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
            responses = []
            id_str = ",".join(map(str, pubmed_ids))
            response = requests.get(f"{url}{id_str}&api_key=691e69e351f883545c190eba28c4b792b308")
            assert response.status_code == 200
            response = response.json()
            return response['result']

        except Exception as e:
            print(e)
            print("Had an error will try again in thirty minutes!!")
            time.sleep(1800)


# In[3]:


pmid_df = pd.read_csv(
    "output/pmid.tsv", 
    sep="\t",
    names=["pmid"]
)
print(pmid_df.shape)
pmid_df.head()


# In[4]:


if Path("output/pmid_to_pub_date.tsv").exists():
    # Start from checkpoint incase something goes wrong
    parsed_ids = pd.read_csv("output/pmid_to_pub_date.tsv", sep="\t")
    parsed_ids_set = set(parsed_ids.pmid.tolist())
    remaining_ids = set(pmid_df.pmid.tolist()) - parsed_ids_set
    pmid_df = pd.DataFrame(remaining_ids, columns=["pmid"])
    print(pmid_df.shape)
    
    outfile = open("output/pmid_to_pub_date.tsv", "a")
    writer = csv.DictWriter(
        outfile, 
        fieldnames=["pmid", "pub_date"],
        delimiter="\t",
    )
    
else:
    # Start from scratch
    outfile = open("output/pmid_to_pub_date.tsv", "w")
    writer = csv.DictWriter(
        outfile, 
        fieldnames=["pmid", "pub_date"],
        delimiter="\t",
    )
    writer.writeheader()


# In[5]:


chunk_size = 100
maxsize = pmid_df.shape[0]


# In[6]:


for batch in tqdm.tqdm(range(0, maxsize, chunk_size)):
    doc_ids = (
        pmid_df
        .sort_values("pmid")
        .pmid.values[batch:batch+chunk_size]
    )
    
    records = call_entrez(doc_ids)

    # Parse the initial query
    for record in records['uids']:
        if 'pubdate' not in records[record]:
            writer.writerow({
                "pmid": records[record]['uid'], 
                "pub_date": ''
            })
        else:
            writer.writerow({
                "pmid": records[record]['uid'], 
                "pub_date": records[record]['pubdate']
            })


# In[7]:


outfile.close()

