"""
This module organizes all output data each decade. In other words, it 
concatenates all the 'similarity_scores_{str(year)}-{str(year+9)}.tsv' files into 
a single file ('total_data.tsv') and adds a column with the appropriate decade for each row. 
The concatenated data will be used to generate time plots in plots.R. 
"""
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":
    base = os.path.abspath(os.getcwd())
    df = pd.DataFrame(columns = ["disease_name", "disease", "gene_name", "gene", "class", "score", "year"])

    do_mesh_df = pd.read_csv("inputs/DO-slim-to-mesh.tsv", sep="\t")
    diseases = dict(zip(
        "MESH:"+do_mesh_df.mesh_id,
        do_mesh_df.doid_name
    ))

    gene_disease_df = pd.read_csv("inputs/hetnet_gene_disease_pairs.tsv", sep="\t")
    genes = dict(zip(
        gene_disease_df.entrez_gene_id,
        gene_disease_df.gene_symbol
    ))

    years = [1971, 1981, 1991, 2001, 2011]
    for filename in os.listdir(os.path.join(base, "outputs/decades/")):
        for year in years:
            if f"similarity_scores_{str(year)}-{str(year+9)}" in filename:
                print(os.path.join(base, filename))
                curr_df = pd.read_csv(
                    os.path.join(base, "outputs/decades/", filename),
                    sep="\t"
                )
                curr_df["year"] = f"{str(year)}-{str(year+9)}"
                curr_df["disease_name"] = curr_df["disease"].replace(diseases)
                curr_df["gene_name"] = curr_df["gene"].replace(genes)
                curr_df = curr_df.sort_values(by=["score"], ascending=False)
                df = df.append(curr_df)

    df.to_csv("outputs/decades/total_data.tsv", sep="\t", index=False)
