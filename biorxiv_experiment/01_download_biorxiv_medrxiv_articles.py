# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Download all the Current Preprints

# Download all the preprints in biorxiv and medrxiv so we can detect which tokens are changing through time.

# +
from pathlib import Path
import re
import subprocess
import zipfile

import pandas as pd
import plydata as ply
import tqdm
# -

# # BioRxiv

biorxiv_doc_hash_mapper_df = pd.read_csv(
    "output/biorxiv_doc_hash_mapper_updated.tsv", sep="\t"
)
biorxiv_doc_hash_mapper_df.head()

already_parsed_folders = set(
    [
        Path(hash_file).parts[1:2][0]
        for hash_file in set(biorxiv_doc_hash_mapper_df.hash.tolist())
        if Path(hash_file).parts[1:2][0] != "January_2020"
    ]
)
for folder in already_parsed_folders:
    Path(f"output/temp_batch_holder/{str(folder)}").mkdir(exist_ok=True, parents=True)

doc_hash_mapper = set(
    Path(hash_file).name for hash_file in biorxiv_doc_hash_mapper_df.hash.tolist()
)
new_doc_ids = set()
new_doc_hash_rows = []

biorxiv_file_log = Path("output/biorxiv_batch_dirs.txt")
if not biorxiv_file_log.exists():
    print("Please execute the following system commands line before continuing:")
    print(
        "s3cmd ls s3://biorxiv-src-monthly/Back_Content/* --requester-pays --recursive > output/biorxiv_batch_dirs_1.log"
    )
    print(
        "s3cmd ls s3://biorxiv-src-monthly/Current_Content/* --requester-pays --recursive > output/biorxiv_batch_dirs_2.log"
    )
    print("once the commands have finished please merge the files into one")
else:
    with open(str(biorxiv_file_log), "r") as infile:
        # doc_downloaded
        for biorxiv_batch in infile:

            biorxiv_batch = biorxiv_batch.strip()
            biorxiv_batch_name = Path(biorxiv_batch).parts[2:]
            batch_folder = Path(
                f"output/temp_batch_holder/biorxiv/{biorxiv_batch_name[1]}"
            )

            if not batch_folder.exists():
                print(f"Downloading {str(biorxiv_batch)}")
                batch_folder.mkdir(exist_ok=True, parents=True)
                result = subprocess.check_output(
                    f"s3cmd get {biorxiv_batch}* {batch_folder}/. --requester-pays",
                    shell=True,
                )

            for preprint_zipfile in tqdm.tqdm(batch_folder.rglob("*meca")):
                # Remove if already parsed
                if preprint_zipfile.name in doc_hash_mapper:
                    preprint_zipfile.unlink()
                    continue

                try:
                    with zipfile.ZipFile(preprint_zipfile) as infile:
                        filename = list(
                            filter(
                                lambda x: str(Path(x).parent) == "content"
                                and Path(x).suffix == ".xml",
                                infile.namelist(),
                            )
                        )

                        version_count = 1
                        output_file_name = f"{Path(filename[0]).name}_v{version_count}"
                        while output_file_name in new_doc_ids:
                            version_count += 1
                            output_file_name = (
                                f"{Path(filename[0]).stem}_v{version_count}"
                            )

                        new_doc_ids.add(output_file_name)

                        article_file = Path(
                            f"output/biorxiv_medrxiv_dump/biorxiv/{output_file_name}.xml"
                        )

                        new_doc_hash_rows.append(
                            {
                                "hash": f"{'/'.join(biorxiv_batch_name)}/{preprint_zipfile.name}",
                                "doc_number": str(article_file.name),
                            }
                        )

                        # Write new preprint to file
                        with infile.open(str(filename[0])) as doc_xml, open(
                            str(article_file), "w"
                        ) as outfile:
                            outfile.write(doc_xml.read().decode("utf-8"))

                        preprint_zipfile.unlink()
                except Exception as e:
                    print(e)
                    print(f"Email biorxiv about this hash: {preprint_zipfile}")
                    preprint_zipfile.unlink()

if len(new_doc_hash_rows) > 0:
    (
        biorxiv_doc_hash_mapper_df
        >> ply.call(".append", pd.DataFrame.from_records(new_doc_hash_rows))
        >> ply.call(".reset_index")
        >> ply.select("-index")
        >> ply.call(
            ".to_csv",
            "output/biorxiv_doc_hash_mapper_updated.tsv",
            sep="\t",
            index=False,
        )
    )

# # MedRxiv

if Path("output/medrxiv_doc_hash_mapper_updated.tsv").exists():
    medrxiv_doc_hash_mapper_df = pd.read_csv(
        "output/medrxiv_doc_hash_mapper_updated.tsv", sep="\t"
    )
else:
    medrxiv_doc_hash_mapper_df = pd.DataFrame([], columns=["hash", "doc_number"])
medrxiv_doc_hash_mapper_df.head()

already_parsed_folders = set(
    [
        Path(hash_file).parts[1:2][0]
        for hash_file in set(medrxiv_doc_hash_mapper_df.hash.tolist())
    ]
)
for folder in already_parsed_folders:
    Path(f"output/temp_batch_holder/medrxiv/{str(folder)}").mkdir(
        exist_ok=True, parents=True
    )

doc_hash_mapper = set(
    Path(hash_file).name for hash_file in medrxiv_doc_hash_mapper_df.hash.tolist()
)
new_doc_ids = set()
new_doc_hash_rows = []

medrxiv_file_log = Path("output/medrxiv_batch_dirs.txt")
if not medrxiv_file_log.exists():
    print("Please execute the following system commands line before continuing:")
    print(
        "s3cmd ls s3://medrxiv-src-monthly/Back_Content/* --requester-pays --recursive > output/medrxiv_batch_dirs_1.txt"
    )
    print(
        "s3cmd ls s3://medrxiv-src-monthly/Current_Content/* --requester-pays --recursive > output/medrxiv_batch_dirs_2.txt"
    )
    print("once the commands have finished please merge the files into one")
else:
    with open(str(medrxiv_file_log), "r") as infile:
        # doc_downloaded
        for medrxiv_batch in infile:

            medrxiv_batch = medrxiv_batch.strip()
            medrxiv_batch_name = Path(medrxiv_batch).parts[2:]
            batch_folder = Path(
                f"output/temp_batch_holder/medrxiv/{medrxiv_batch_name[1]}"
            )

            if not batch_folder.exists():
                print(f"Downloading {str(medrxiv_batch)}")
                batch_folder.mkdir(exist_ok=True, parents=True)
                result = subprocess.check_output(
                    f"s3cmd get {medrxiv_batch}* {batch_folder}/. --requester-pays",
                    shell=True,
                )

            for preprint_zipfile in tqdm.tqdm(batch_folder.rglob("*meca")):
                # Remove if already parsed
                if preprint_zipfile.name in doc_hash_mapper:
                    preprint_zipfile.unlink()
                    continue

                try:
                    with zipfile.ZipFile(preprint_zipfile) as infile:
                        filename = list(
                            filter(
                                lambda x: str(Path(x).parent) == "content"
                                and Path(x).suffix == ".xml",
                                infile.namelist(),
                            )
                        )

                        version_count = 1
                        output_file_name = f"{Path(filename[0]).stem}_v{version_count}"
                        while output_file_name in new_doc_ids:
                            version_count += 1
                            output_file_name = (
                                f"{Path(filename[0]).stem}_v{version_count}"
                            )

                        new_doc_ids.add(output_file_name)

                        article_file = Path(
                            f"output/biorxiv_medrxiv_dump/medrxiv/{output_file_name}.xml"
                        )

                        new_doc_hash_rows.append(
                            {
                                "hash": f"{'/'.join(medrxiv_batch_name)}/{preprint_zipfile.name}",
                                "doc_number": str(article_file.name),
                            }
                        )

                        # Write new preprint to file
                        with infile.open(str(filename[0])) as doc_xml, open(
                            str(article_file), "w"
                        ) as outfile:
                            outfile.write(doc_xml.read().decode("utf-8"))

                        preprint_zipfile.unlink()
                except Exception as e:
                    print(e)
                    print(f"Email medrxiv about this hash: {preprint_zipfile}")
                    preprint_zipfile.unlink()

if len(new_doc_hash_rows) > 0:
    (
        medrxiv_doc_hash_mapper_df
        >> ply.call(".append", pd.DataFrame.from_records(new_doc_hash_rows))
        >> ply.call(".reset_index")
        >> ply.select("-index")
        >> ply.call(
            ".to_csv",
            "output/medrxiv_doc_hash_mapper_updated.tsv",
            sep="\t",
            index=False,
        )
    )
