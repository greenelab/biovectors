# Biovectors

## Citation

BioRxiv Citation - (TBD)

## About

This repository is designed to detect semantic changes within biomedical literature using Pubtator Central and bioRxiv.
It contains the scripts to train word2vec models along with downloading and extracting documents from Pubtator Central and bioRxiv.

## Data Availability
Data for each figure in our manuscript can be found in our [FIGURE_DATA_SOURCE.md](FIGURE_DATA_SOURCE.md) file.
This file contains relative links for each data source used to generate each piece of the figure panel.


## Directory Structure
| Folder/file | Description |
| --- | --- |
| [alignment_check_experiment](alignment_check_experiment) | This folder contains code to validate word2vec model alignment. |
| [bayesian_changepoint_detection_experiment](bayesian_changepoint_detection_experiment) | This folder contains all experiments that are related to changepoint detection.  |
| [biovectors_modules](biovectors_modules) | This folder contains supporting functions that other notebooks in this repository will use. |
| [biorxiv_experiment](biorxiv_experiment) | This folder contains code for downloading and extracting preprints from biorxiv and medrxiv. |
| [concept_mapper_experiment](concept_mapper_experiment) | This folder contains code for creating an ID to concepts mapper. |
| [data_analysis_pipeline](data_analysis_pipeline) | This folder contains code for running the changepoint detection pipeline. Each file is in script form. |
| [exploratory_data_analysis](exploratory_data_analysis) | This folder contains code that analyzes the Pubtator Central dataset. |
| [figure_generation](figure_generation) |  This folder contains code to generate every figure for the manuscript. |
| [full_text_experiments](full_text_experiments) | This folder contains code for funning this project using full text. (depreciated)|
| [gif_animator_experiment](gif_animator_experiment) | This folder explored the idea of creating gifs for changepoints. (depreciated) |
| [multi_model_experiment](multi_model_experiment) | This folder dived deeper into using full text to generate data for this experiment. (depreciated as Pubtator updated and made results outdated).
| [multi_model_revamp](multi_model_revamp) |  This folder dived deeper into using full text to generate data for this experiment post Pubtator update. |
| [pubmed_timestamp_experiment](pubmed_timestamp_experiment) | This folder extract timepoints for each pubtator document for a rotation student. (depreciated)|
| [rotation_scripts](rotation_scripts) | The scripts used for a rotation project.|
| [timeline_shifts_visualization_experiment](timeline_shifts_visualization_experiment) | This folder examines utilizing umap to visualize how words shift through time. |
| [word2vec_decade_experiment](word2vec_decade_experiment) | This folder contains scratch work for analyzing changes through time. |
| [word2vec_prediction_experiment](word2vec_prediction_experiment) | This was a rotation project for seeing if word2vec can predict biomedical relationships. |
| [environment.yml](environment.yml) | This file contains the necessary packages this repository uses. |
| [install.sh](install.sh) | A bash script to set up the repository. |
| [setup.py](setup.py) | This file sets up the annorxiver modules to be used as a regular python package. |


## Installation Instructions

Biovectors uses [conda](http://conda.pydata.org/docs/intro.html) as a python package manager.
Before moving on to the instructions below, please make sure to have it installed.
[Download conda here!!](https://docs.conda.io/en/latest/miniconda.html)

Once everything has been installed, type following command in the terminal:

```bash
bash install.sh
```
_Note_:
There is a bash command within the install.sh that only works on unix systems (```source ~/anaconda3/etc/profile.d/conda.sh```).
If you are on windows (and possibly mac), you should remove that line or execute each command individually.
Alternatively for windows users please refer to these [installation instructions](https://itsfoss.com/install-bash-on-windows/) to enable bash on windows.

You can activate the environment by using the following command:

```bash
conda activate biovectors
```

## License

Please look at [LICENSE.md](LICENSE.md).
