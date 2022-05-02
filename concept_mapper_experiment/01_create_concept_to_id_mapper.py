# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1+dev
#   kernelspec:
#     display_name: Python [conda env:biovectors]
#     language: python
#     name: conda-env-biovectors-py
# ---

# # Extract all concept IDs for Pubtator Central

# +
import gzip
import os
from pathlib import Path
import re

import pandas as pd
import plydata as ply
import tqdm
# -

Path("output/intermediate_files").mkdir(exist_ok=True, parents=True)

# # Extract MESH IDs

mesh_chemical_url = (
    "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/c2022.bin"
)
mesh_disease_url = (
    "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/asciimesh/d2022.bin"
)

# +
chemical_file = f"output/intermediate_files/{Path(mesh_chemical_url).name}"
if not Path(chemical_file).exists():
    os.system(f"wget {mesh_chemical_url} -O {chemical_file}")

disease_file = f"output/intermediate_files/{Path(mesh_disease_url).name}"
if not Path(disease_file).exists():
    os.system(f"wget {mesh_disease_url} -O {disease_file}")
# -

mesh_map_file = Path("output/mesh_headings_id_mapper.tsv")
if not mesh_map_file.exists():
    with open(disease_file, "r") as disease_infile, open(
        chemical_file, "r"
    ) as chem_infile:
        data_rows = []
        for line in tqdm.tqdm(disease_infile):
            line = line.strip()
            match_obj = re.search("MH = ", line)

            if match_obj is not None:
                concept = re.sub("MH = ", "", line).lower()

            match_obj = re.search("UI = ", line)
            if match_obj is not None:
                concept_id = re.sub("UI = ", "", line).lower()
                data_rows.append({"mesh_id": concept_id, "mesh_heading": concept})

        for line in tqdm.tqdm(chem_infile):
            line = line.strip()
            match_obj = re.search("MH = ", line)

            if match_obj is not None:
                concept = re.sub("MH = ", "", line).lower()

            match_obj = re.search("UI = ", line)
            if match_obj is not None:
                concept_id = re.sub("UI = ", "", line).lower()
                data_rows.append({"mesh_id": concept_id, "mesh_heading": concept})

    concept_df = pd.DataFrame.from_records(data_rows)
    concept_df >> ply.call("to_csv", str(mesh_map_file), sep="\t", index=False)
else:
    concept_df = pd.read_csv(str(mesh_map_file), sep="\t")
print(concept_df.shape)
concept_df >> ply.slice_rows(10)

# # Extract Species IDs

species_id_url = "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"

species_file = f"output/intermediate_files/{Path(species_id_url).name}"
if not Path(species_file).exists():
    os.system(f"wget {species_id_url} -O {species_file}")
    Path("output/intermediate_files/species_temp").mkdir(exist_ok=True, parents=True)
    os.system(
        f"gunzip -c {species_file} | tar xf - -C output/intermediate_files/species_temp"
    )

# +
species_map_file = Path("output/species_id_map.tsv")
if not species_map_file.exists():
    with open(
        "output/intermediate_files/species_temp/names.dmp", "r"
    ) as species_infile:
        data_rows = []
        for idx, line in enumerate(species_infile):
            fields = re.sub("\t", "", line.strip()).split("|")

            if fields[3] == "scientific name":
                data_rows.append(
                    {"species_id": int(fields[0]), "species_name": fields[1].lower()}
                )

        species_df = pd.DataFrame.from_records(data_rows)
        species_df >> ply.call("to_csv", str(species_map_file), sep="\t", index=False)
else:
    species_df = pd.read_csv(str(species_map_file), sep="\t")

print(species_df.shape)
species_df >> ply.slice_rows(10)
# -

# # Extract Gene IDS

gene_id_url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/All_Data.gene_info.gz"

gene_file = Path(f"output/intermediate_files/{Path(gene_id_url).name}")
if not gene_file.exists():
    os.system(f"wget {gene_id_url} -O {gene_file}")

gene_map_file = Path("output/gene_id_map.tsv")
if not gene_map_file.exists():
    with gzip.open(str(gene_file), "rt") as infile:
        data_rows = []
        for idx, line in tqdm.tqdm(enumerate(infile)):
            if idx == 0:
                continue

            gene_row = line.split("\t")
            data_rows.append(
                {
                    "tax_id": gene_row[0],
                    "gene_id": gene_row[1],
                    "gene_symbol": gene_row[2],
                }
            )

        gene_df = pd.DataFrame.from_records(data_rows)
        gene_df >> ply.call("to_csv", str(gene_map_file), sep="\t", index=False)
else:
    gene_df = pd.read_csv(str(gene_map_file), sep="\t")
print(gene_df.shape)
gene_df >> ply.slice_rows(10)

# # Extract Cell Line IDs

celline_id_url = "https://ftp.expasy.org/databases/cellosaurus/cellosaurus.obo"

celline_file = f"output/intermediate_files/{Path(celline_id_url).name}"
if not Path(celline_file).exists():
    os.system(f"wget {celline_id_url} -O {celline_file}")

# +
celline_map_file = Path("output/celline_id_map.tsv")
if not celline_map_file.exists():
    with open(celline_file, "r") as celline_infile:
        data_rows = []
        for idx, line in tqdm.tqdm(enumerate(celline_infile)):
            line = line.strip()
            if "id:" in line:
                entry = dict(celline_id=line.split(" ")[1])

            if "name:" in line:
                entry["celline_name"] = line.split(" ")[1]
                data_rows.append(entry)

        celline_df = pd.DataFrame.from_records(data_rows)
        celline_df >> ply.call("to_csv", str(celline_map_file), sep="\t", index=False)
else:
    celline_df = pd.read_csv(str(celline_map_file), sep="\t")

print(celline_df.shape)
celline_df >> ply.slice_rows(10)
# -

# # Merge all ids into one

total_concept_file = Path("output/all_concept_ids.tsv.xz")
if not total_concept_file.exists():
    total_concepts_df = pd.concat(
        [
            (
                concept_df
                >> ply.define(
                    concept_id="mesh_id.apply(lambda x: 'mesh_'+x.lower())",
                    concept="mesh_heading",
                )
                >> ply.select("concept_id", "concept")
            ),
            (
                species_df
                >> ply.define(
                    concept_id="species_id.apply(lambda x: 'species_'+str(x))",
                    concept="species_name",
                )
                >> ply.select("concept_id", "concept")
            ),
            (
                gene_df
                >> ply.define(
                    concept_id="gene_id.apply(lambda x: 'gene_'+str(x))",
                    concept="gene_symbol",
                )
                >> ply.select("concept_id", "concept")
            ),
            (
                celline_df
                >> ply.define(
                    concept_id="celline_id.apply(lambda x: 'cellline_'+str(x))",
                    concept="celline_name",
                )
                >> ply.select("concept_id", "concept")
            ),
        ]
    )
    total_concepts_df >> ply.call(
        "to_csv", str(total_concept_file), sep="\t", index=False, compress="xz"
    )
else:
    total_concepts_df = pd.read_csv(str(total_concept_file), sep="\t")
total_concepts_df >> ply.sample_n(20, random_state=100)
