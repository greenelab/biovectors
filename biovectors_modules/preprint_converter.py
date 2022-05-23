import html

import lxml.etree as ET
import re


def extract_paragraph_text(p_tag_list):
    for paragraph_section in p_tag_list:
        text = "".join(paragraph_section.xpath("text()"))
        cleaned_text =re.sub("&lt;|&gt;", "", text)  # noqa: E225
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub("&", "", cleaned_text)
        yield cleaned_text


def traverse_all_the_sections(section, passage_builder=None):
    # initalize list at the start
    if passage_builder is None:
        passage_builder = list()

    if len(section) == 0:
        return passage_builder
    else:
        for sub_section in section:
            passage_title = list(
                extract_paragraph_text(
                    sub_section.xpath(
                        "title[not(parent::title)='fig' and not(parent::title)='table-wrap']"
                    )
                )
            )

            if len(passage_title) == 0:
                passage_title = [""]

            passage_text = list(
                extract_paragraph_text(
                    sub_section.xpath(
                        "p[not(parent::p)='fig' and not(parent::p)='table-wrap']"
                    )
                )
            )

            if len(passage_text) == 0:
                passage_title = [""]

            passage_builder.append("\n".join(passage_title + passage_text))
            final_passage_builder = traverse_all_the_sections(
                sub_section.xpath("sec"), passage_builder
            )
        return final_passage_builder


def create_passage_element(infon_top_section, offset, text_generator, **extra_info):
    passages = []
    starting_offset = offset
    for text in text_generator:
        passage_element = ET.Element("passage")
        infon_element = ET.Element("infon", key="section_type")
        infon_element.text = infon_top_section
        offset_element = ET.Element("offset")
        offset_element.text = str(starting_offset)
        text_element = ET.Element("text")
        text_element.text = text

        for key in extra_info:
            infon_element = ET.Element("infon", key=key)
            infon_element.text = extra_info[key]

        starting_offset += len(text)
        passage_element.extend([infon_element, offset_element, text_element])
        passages.append(passage_element)
    return passages, starting_offset


def convert_to_bioc(preprint_root_tree, repository):
    output_root = ET.Element("collection")
    source = ET.Element("source")
    source.text = repository.upper()
    date = ET.Element("date")
    date.set("date-type", "accepted")
    date.text = "-".join(
        preprint_root_tree.xpath(
            "/article/front/article-meta/history/date[@date-type='accepted']/*/text()"
        )
    )
    key = ET.Element("key")
    key.text = "biorxiv_medrxiv.key"

    document_collection = ET.Element("document")
    doc_id = ET.Element("id")
    doc_id.text = preprint_root_tree.xpath("/article/front/article-meta/article-id/text()")[0]
    document_collection.append(doc_id)

    # Extract the article title
    running_offset = 0
    article_title = preprint_root_tree.xpath(
        "/article/front/article-meta/title-group/article-title"
    )[0]
    article_title_text = extract_paragraph_text([article_title])
    passages_for_doc_collection, running_offset = create_passage_element(
        "Title", running_offset, article_title_text
    )
    document_collection.extend(passages_for_doc_collection)

    # Extract the abstract
    for abstract in preprint_root_tree.xpath("/article/front/article-meta/abstract"):
        abs_title = abstract.xpath("title/text()")
        if len(abs_title) == 0:
            for abs_section in abstract.xpath("sec"):
                abs_section_title = abs_section.xpath("title/text()")
                abs_section_title = (
                    ""
                    if len(abs_section_title) == 0
                    else abs_section_title[0].lower().capitalize()
                )

                abs_section_passage_text = extract_paragraph_text(abs_section.xpath("p"))
                passages_for_doc_collection, running_offset = create_passage_element(
                    abs_section_title, running_offset + 1, abs_section_passage_text
                )
                document_collection.extend(passages_for_doc_collection)

                sub_abs_passages_text = traverse_all_the_sections(abs_section.xpath("sec"))
                passages_for_doc_collection, running_offset = create_passage_element(
                    abs_section_title, running_offset + 1, sub_abs_passages_text
                )
                document_collection.extend(passages_for_doc_collection)

        else:
            abs_title = (
                ""
                if len(abs_title) == 0
                else abs_title[0].lower().capitalize()
            )
            abs_text = extract_paragraph_text(abstract.xpath("p"))
            passages_for_doc_collection, running_offset = create_passage_element(
                abs_title.lower().capitalize(), running_offset + 1, abs_text
            )
            document_collection.extend(passages_for_doc_collection)

    # Iterate all the main text for each document
    for top_section in preprint_root_tree.xpath("/article/body/sec"):
        top_section_title = top_section.xpath("title/text()")
        top_section_title = (
            ""
            if len(top_section_title) == 0
            else top_section_title[0].lower().capitalize()
        )

        top_section_passage_text = extract_paragraph_text(top_section.xpath("p"))
        passages_for_doc_collection, running_offset = create_passage_element(
            top_section_title, running_offset + 1, top_section_passage_text
        )
        document_collection.extend(passages_for_doc_collection)

        sub_passages_text = traverse_all_the_sections(top_section.xpath("sec"))
        passages_for_doc_collection, running_offset = create_passage_element(
            top_section_title, running_offset + 1, sub_passages_text
        )
        document_collection.extend(passages_for_doc_collection)

    for figures in preprint_root_tree.xpath("//fig"):
        figure_label = figures.xpath("label/text()")
        figure_label = "" if len(figure_label) == 0 else figure_label[0]

        figure_caption_text = extract_paragraph_text(
            figures.xpath("caption/title|caption/p")
        )
        passages_for_doc_collection, running_offset = create_passage_element(
            "Figure", running_offset + 1, figure_caption_text, figure_label=figure_label
        )
        document_collection.extend(passages_for_doc_collection)

    for tables in preprint_root_tree.xpath("//table-wrap"):
        table_label = tables.xpath("label/text()")
        table_label = "" if len(table_label) == 0 else table_label[0]

        tables_caption_text = extract_paragraph_text(
            tables.xpath("caption/title|caption/p")
        )
        passages_for_doc_collection, running_offset = create_passage_element(
            "Table", running_offset + 1, tables_caption_text, tabel_label=table_label
        )
        document_collection.extend(passages_for_doc_collection)

    output_root.extend([source, date, key, document_collection])
    return output_root
