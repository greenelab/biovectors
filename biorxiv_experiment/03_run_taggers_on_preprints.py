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

# # Tag Preprints with Pubtator Central's Taggers

# This notebook runs Pubtator's specific taggers on the converted preprint files.
# These taggers are mainly written in java (except for GnormPlus that has a perl option).
# I suggest only using the java version as it was the easiest to set up and get running.
# Also, as a side note some of these taggers will require a bit of hitting your head against the wall as one of their external programs will need you to recompile to run.
#
# | Tagger | URL |
# | --- | --- |
# | Tagger One | https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/taggerone/ |
# | GNormPlus  | https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/ |

# +
from pathlib import Path

import os
import subprocess
# -

converted_preprint_folders = sorted(
    list(Path("output/converted_docs").glob("*"), key=lambda x: int(x.stem))
)

preprint_folders = [folder.absolute() for folder in converted_preprint_folders]
preprint_folders

log_output_folder = Path("output/tagging_logs/").absolute()

# # GNormPlus Tagger

gnorm_tagger_path = Path("../../tagger_test/GNormPlusJava/").absolute()
gnorm_output_folder = Path("output/gnormplus_tags/").absolute()

os.chdir(str(gnorm_tagger_path))
for preprint_folder in preprint_folders:
    print(preprint_folder)
    f = open(f"{log_output_folder}/{preprint_folder.stem}_tagger.log", "w")
    p = subprocess.call(
        [
            f"sh run_gnormplus.sh {str(preprint_folder)} {gnorm_output_folder}/{preprint_folder.stem}"
        ],
        stdout=f,
        stderr=f,
        shell=True,
    )
    if p != 0:
        break

# # Tagger One Tagger

tagger_one_path = Path("../TaggerOne-0.2.1/").absolute()
tagger_one_output_folder = Path("output/tagger_one_tags").absolute()

os.chdir(str(tagger_one_path))
for preprint_folder in preprint_folders:
    for preprint_file in preprint_folder.rglob("*xml"):

        if Path(f"{tagger_one_output_folder}/{preprint_file.name}").exists():
            continue

        f = open(f"{log_output_folder}/{preprint_folder.stem}_tagger.log", "w")
        p = subprocess.call(
            [
                f"sh ProcessText.sh BioC output/model_DISE.bin {str(preprint_file)} {str(tagger_one_output_folder)}/{str(preprint_file.name)}"
            ],
            stdout=f,
            stderr=f,
            shell=True,
        )
