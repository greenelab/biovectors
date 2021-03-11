import gzip
from multiprocessing import Process, Manager
from pathlib import Path
import random
import tarfile
from threading import Thread

import lxml.etree as ET
import pandas as pd
import spacy

random.seed(100)


class PubtatorTarIterator:
    """
    Accesses Abstracts and Full text without extracting batches from the
    tar gz Pubtator Central created
    """

    def __init__(self, tarfile_name):
        self.pubtator_tar_directory = tarfile.open(tarfile_name)

    def __iter__(self):
        # Iterate through documents within each Pubtator Batch
        while True:
            try:
                pubtator_batch = self.pubtator_tar_directory.next()

                if pubtator_batch is None:
                    break

                doc_iterator = ET.iterparse(
                    self.pubtator_tar_directory.extractfile(pubtator_batch),
                    tag="document",
                    recover=True,
                )

                # Yield the individual document objects
                for event, doc_obj in doc_iterator:
                    yield doc_obj

                    doc_obj.clear()

            except tarfile.ReadError:
                print("Parsing the archive reached an error. Breaking out the loop")
                break


class PubMedSentencesIterator:
    """
    Extracts title + abstracts from Pubtator data. Replaces any entity instance
    (i.e. 'gene', 'disease)
    with their respective identifier type (i.e. 'Entrez gene id', 'MESH id').
    """

    def __init__(
        self,
        pubmed_tarfiles,
        batch_skipper=None,
        year_filter=None,
        section_filter=None,
        return_year=False,
        tag_entities=True,
        jobs=4,
    ):
        self.pubmed_tarfiles = pubmed_tarfiles

        if batch_skipper is None:
            batch_skipper = []

        self.batch_skipper = batch_skipper

        if year_filter is None:
            year_filter = []

        self.year_filter = year_filter

        if section_filter is None:
            section_filter = ["TITLE", "ABSTRACT"]

        self.section_filter = section_filter

        self.tag_entities = tag_entities
        self.return_year = return_year
        self.jobs = jobs

    def _feed_in_pubmed_objs(self, pubmed_obj_queue):
        for pubtator_batch in self.pubmed_tarfiles:

            # Skip if batch is in skipper
            if Path(pubtator_batch).name in self.batch_skipper:
                continue

            for doc_obj in PubtatorTarIterator(pubtator_batch):
                pubmed_obj_queue.put(ET.tostring(doc_obj))

                doc_obj.clear()

        # Tell the jobs to end from the feeding thread
        for job in range(self.jobs):
            pubmed_obj_queue.put(None)  # poison pill to end the processes

    def _process_document_objs(self, pubmed_obj_queue, sen_queue):
        nlp = spacy.load("en_core_web_sm")
        while True:
            doc_obj = pubmed_obj_queue.get()

            if doc_obj is None:
                break

            doc_obj = ET.fromstring(doc_obj)
            year = doc_obj.xpath(
                "passage[contains(infon[@key='section_type']/text(), 'TITLE')]/infon[@key='year']/text()"
            )

            # Skip if year not in the filter
            # Given that user wants to filter years out
            if len(year) == 0:
                continue

            if len(self.year_filter) > 0 and int(year[0]) not in self.year_filter:
                continue

            for passage in doc_obj.xpath("passage"):
                section = passage.xpath("infon[@key='section_type']/text()")

                if section[0] not in self.section_filter:
                    continue

                passage_text = passage.xpath("text/text()")

                if len(passage_text) < 1:
                    continue

                passage_text = passage_text[0]

                passage_offset = passage.xpath("offset/text()")[0]
                current_pos = 0
                yield_text = ""

                if self.tag_entities:
                    sorted_passages = sorted(
                        passage.xpath("annotation"),
                        key=lambda x: int(x.xpath("location")[0].attrib["offset"]),
                    )

                    for annotation in sorted_passages:
                        annot_identifier = annotation.xpath(
                            "infon[@key='identifier']/text()"
                        )

                        if len(annot_identifier) == 0 or annot_identifier[0] == "-":
                            continue

                        annot_type = annotation.xpath("infon[@key='type']/text()")
                        location = annotation.xpath("location")

                        # replace string with identifier
                        entity_start = int(location[0].attrib["offset"]) - int(
                            passage_offset
                        )
                        entity_end = entity_start + int(location[0].attrib["length"])
                        replacement_str = f" {annot_type[0].upper()}_{annot_identifier[0].replace(':','_')} "
                        yield_text += (
                            passage_text[current_pos:entity_start].lower()
                            + replacement_str
                        )
                        current_pos = entity_end

                yield_text += passage_text[current_pos:].lower()
                analyzed_text = nlp(yield_text)

                sen_queue.put((int(year[0]), list(map(str, analyzed_text))))

        sen_queue.put(None)  # Poison pill for sentence feeder

    def __iter__(self):
        # print("Getting sentences...")

        # randomly shuffle sentences
        random.shuffle(self.pubmed_tarfiles)
        finished_job_count = 0

        with Manager() as m:
            # Set up the Queue
            pubmed_obj_queue = m.JoinableQueue(50000)
            sen_queue = m.JoinableQueue(50000)

            # Start the document object feeder
            t = Thread(target=self._feed_in_pubmed_objs, args=(pubmed_obj_queue,))
            t.start()

            # Start the jobs
            runnable_jobs = []
            for job in range(self.jobs):
                p = Process(
                    target=self._process_document_objs,
                    args=(pubmed_obj_queue, sen_queue),
                )
                runnable_jobs.append(p)
                p.start()

            # Feed the sentence to the user
            while True:
                sentence = sen_queue.get()

                # Count the number of jobs being finished.
                # If total equals num of jobs launched
                # then break out of while loop to finish iteration
                if sentence is None:
                    finished_job_count += 1

                    if finished_job_count == self.jobs:
                        break

                    continue

                if self.return_year:
                    yield sentence[0], sentence[1]

                else:
                    yield sentence[1]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_gene_disease_pairs(gene_disease_filename, do_mesh_filename):
    """
    Extracts hetionet gene-disease pairs and generates negative pairs by randomizing positive pairs.
    @param gene_disease_filename: file containing hetionet gene disease pairs
    @param do_mesh_filename: file containing corresponding doid and mesh ids (because hetnet
        contains only doids)
    @return positive and negative gene-disease pairs
    """
    random.seed(100)  # reproducibility
    gene_disease_df = pd.read_csv(gene_disease_filename, sep="\t")
    do_mesh_df = pd.read_csv(do_mesh_filename, sep="\t")

    # create doid-mesh list
    do_mesh_pairs = dict(zip(do_mesh_df.doid_code, "MESH:" + do_mesh_df.mesh_id))
    gene_disease_df["mesh_id"] = gene_disease_df["doid_id"].replace(do_mesh_pairs)
    # remove rows that don't have a DOID-MESH id mapping
    # gene_disease_df = gene_disease_df.query("~mesh_id.str.contains('DOID:')")
    gene_disease_df = gene_disease_df[~gene_disease_df.mesh_id.str.contains("DOID:")]
    # get positive pairs
    positive_pairs = gene_disease_df[["mesh_id", "entrez_gene_id"]].values.tolist()

    # randomize pairings to create negative pairs
    gene_disease_df["random_gene"] = random.sample(
        gene_disease_df["entrez_gene_id"].values.tolist(),
        len(gene_disease_df["entrez_gene_id"].values.tolist()),
    )
    randomized_pairs = gene_disease_df[["mesh_id", "random_gene"]].values.tolist()
    negative_pairs = []
    for pair in random.sample(randomized_pairs, len(randomized_pairs)):
        if pair not in positive_pairs:
            negative_pairs.append(pair)

    # append class to each pair
    for pair in positive_pairs:
        pair.append(1)
    for pair in negative_pairs:
        pair.append(0)
    gene_disease_pairs = positive_pairs + negative_pairs

    return gene_disease_pairs


def similarity_scores(model, pairs):
    """
    Computes cosine similarity between gene and disease if both exist in the word2vec vocabulary.
    Outputs similarity_scores.tsv.
    @param model: the trained word2vec model
    @param pairs: all gene-disease pairs to be tested
    """
    similarity_scores_df = pd.DataFrame(columns=["disease", "gene", "class", "score"])
    for pair in pairs:
        if all(str(vocab) in model.wv.vocab for vocab in pair[:2]):
            score = model.wv.similarity(str(pair[0]), str(pair[1]))
            new_row = {
                "disease": pair[0],
                "gene": pair[1],
                "class": pair[2],
                "score": score,
            }
            similarity_scores_df = similarity_scores_df.append(
                new_row, ignore_index=True
            )

    return similarity_scores_df
