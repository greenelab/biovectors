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

# # Figure Generation for Manuscript

# +
import os
from pathlib import Path
from IPython.display import Image, display, SVG

from cairosvg import svg2png
from lxml import etree
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import numpy as np
import pandas as pd
from svgutils.compose import Unit
import svgutils.transform as sg
# -

# ## Figure 1

# This figure is designed to show the need for orthogonal procrustes.
# Idea here is that word2vec generates words arbitrarily and direct comparison is meaningless without an alignment factor.
# 1. Panel A - Show umap plot of word models without alignment
# 2. Panel B - Show umap plot of the same word models with alignment
# 3. Panel C - Show same umap plot but with two examples that show intra year variability (show the opacity plot with word such as 'of', 'interleukin-18', 'pandemic')
# 4. Panel D - Systematic comparison of tokens across the years

umap_visualization_path = Path(
    "../multi_model_revamp/output/figure_data_and_figures/alignment_visualization"
)

# +
panel_one = sg.fromfile(umap_visualization_path / "unaligned_2010_plot.svg")

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(50, 50)

# +
panel_two = sg.fromfile(umap_visualization_path / "unaligned_2010_probiotics_plot.svg")

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(670, 50)

# +
panel_three = sg.fromfile(umap_visualization_path / "aligned_2010_probiotics_plot.svg")

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(50, 436)

# +
panel_four = sg.fromfile(umap_visualization_path / "systemic_alignment_metrics.svg")

panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(670, 436)
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(660, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(30, 436, "C", size=30, weight="bold")
panel_four_label = sg.TextElement(660, 436, "D", size=30, weight="bold")

# +
figure_one = sg.SVGFigure(Unit(1350), Unit(880))

figure_one.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_one_label,
        panel_two_label,
        panel_three_label,
        panel_four_label,
    ]
)
# display(SVG(figure_one.to_str()))
# -

# save generated SVG files
figure_one.save("output/Figure_1.svg")
svg2png(bytestring=figure_one.to_str(), write_to="output/Figure_1.png", dpi=300)

# !convert -compress LZW -alpha remove output/Figure_1.png output/Figure_1.tiff
# !mogrify -alpha off output/Figure_1.tiff

# ## Figure 2

# This figure panel shows the validation of the ratio metric

novel_distance_visualization_path = Path(
    "../multi_model_revamp/output/figure_data_and_figures/novel_distance_visualization"
)

# +
panel_one = sg.fromfile(novel_distance_visualization_path / "token_count.svg")

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(50, 50)

# +
panel_two = sg.fromfile(
    novel_distance_visualization_path / "single_model_distance_confirm.svg"
)

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x,panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(660, 50)

# +
panel_three = sg.fromfile(
    novel_distance_visualization_path / "multi_model_distance_confirm.svg"
)

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(70, 464)
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(630, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(30, 464, "C", size=30, weight="bold")

# +
figure_two = sg.SVGFigure(Unit(1280), Unit(930))

figure_two.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_one_label,
        panel_two_label,
        panel_three_label,
    ]
)
# display(SVG(figure_two.to_str()))
# -

# save generated SVG files
figure_two.save("output/Figure_2.svg")
svg2png(bytestring=figure_two.to_str(), write_to="output/Figure_2.png", dpi=600)

# !convert -compress LZW -alpha remove output/Figure_2.png output/Figure_2.tiff
# !mogrify -alpha off output/Figure_2.tiff

# ## Figure 3

# Use this metric through the SCAF approach to detect the specific timepoint at which a word has undergone a drastic shift.
# Idea here is to provide rational that my correction method actually improves detection
# 1. Panel A - Time point example for lung cancer
# 2. Panel B - show wordcloud for this case study and let the users see the shift in semantics
# 3. Could also think about providing the word pandemic or coronavirus as a positive control for this case

published_changepoint_path = Path(
    "../multi_model_revamp/output/figure_data_and_figures/changepoint_amount"
)

preprint_changepoint_path = Path(
    "../biorxiv_experiment/output/figure_data_and_figures/changepoint_amount"
)

# +
panel_one = sg.fromfile(published_changepoint_path / "published_changepoint_amount.svg")

# Convert pt units to pixel units
# Vince's tutorial FTW
panel_one_size = (
    np.round(float(panel_one.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_one.root.attrib["height"][:-2]) * 1.33, 0),
)

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(50, 50)

# +
panel_two = sg.fromfile(preprint_changepoint_path / "preprint_changepoint_amount.svg")

panel_two_size = (
    np.round(float(panel_two.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_two.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(600, 50)

# +
panel_three = sg.fromfile(published_changepoint_path / "cas9_change_metric.svg")

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(50, 500)

# +
panel_four = sg.fromfile(published_changepoint_path / "sars_change_metric.svg")

panel_four_size = (
    np.round(float(panel_four.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_four.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1
scale_y = 1

print(f"original: {panel_four_size}")
print(f"scaled:{(panel_four_size[0]*scale_x, panel_four_size[1]*scale_y)}")

panel_four = panel_four.getroot()
panel_four.scale(x=scale_x, y=scale_y)
panel_four.moveto(600, 500)
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(600, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(30, 500, "C", size=30, weight="bold")
panel_four_label = sg.TextElement(600, 500, "D", size=30, weight="bold")

# +
figure_three = sg.SVGFigure(Unit(1100), Unit(980))

figure_three.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_two,
        panel_three,
        panel_four,
        panel_one_label,
        panel_two_label,
        panel_three_label,
        panel_four_label,
    ]
)
# display(SVG(figure_one.to_str()))
# -

# save generated SVG files
figure_three.save("output/Figure_3.svg")
svg2png(bytestring=figure_three.to_str(), write_to="output/Figure_3.png", dpi=600)

# !convert -compress LZW -alpha remove output/Figure_3.png output/Figure_3.tiff
# !mogrify -alpha off output/Figure_3.tiff
