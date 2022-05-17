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

scale_x = 1.4
scale_y = 1.4

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
scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(954, 50)

# +
panel_three = sg.fromfile(umap_visualization_path / "aligned_2010_probiotics_plot.svg")

panel_three_size = (
    np.round(float(panel_three.root.attrib["width"][:-2]) * 1.33, 0),
    np.round(float(panel_three.root.attrib["height"][:-2]) * 1.33, 0),
)
scale_x = 1.4
scale_y = 1.4

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(50, 631)
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(954, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(30, 631, "C", size=30, weight="bold")

# +
figure_one = sg.SVGFigure(Unit(1650), Unit(1200))

figure_one.append(
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
# display(SVG(figure_one.to_str()))
# -

# save generated SVG files
figure_one.save("output/Figure_1.svg")
svg2png(bytestring=figure_one.to_str(), write_to="output/Figure_1.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_1.png output/Figure_1.tiff"
)
os.system("mogrify -alpha off output/Figure_1.tiff")

# ## Figure 2

# This figure is to show the validation of the metric which is Qst.
# This metric accounts for the intra year instability within each word2vec model.
# 1. Panel - shows the percent difference validation I had for the tokens across the years

novel_distance_visualization_path = Path(
    "../multi_model_revamp/output/figure_data_and_figures/novel_distance_visualization"
)

# +
panel_one = sg.fromfile(
    novel_distance_visualization_path / "percent_difference_2010-2011_plot.svg"
)

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
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")

# +
figure_two = sg.SVGFigure(Unit(900), Unit(480))

figure_two.append(
    [
        etree.Element("rect", {"width": "100%", "height": "100%", "fill": "white"}),
        panel_one,
        panel_one_label,
    ]
)
# display(SVG(figure_two.to_str()))
# -

# save generated SVG files
figure_two.save("output/Figure_2.svg")
svg2png(bytestring=figure_two.to_str(), write_to="output/Figure_2.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_2.png output/Figure_2.tiff"
)
os.system("mogrify -alpha off output/Figure_2.tiff")

# ## Figure 3

# Use this metric through the SCAF approach to detect the specific timepoint at which a word has undergone a drastic shift.
# Idea here is to provide rational that my correction method actually improves detection
# 1. Panel A - Time point example for lung cancer
# 2. Panel B - show wordcloud for this case study and let the users see the shift in semantics
# 3. Could also think about providing the word pandemic or coronavirus as a positive control for this case

neighbor_table_path = Path(
    "../multi_model_revamp/output/figure_data_and_figures/neighbor_tables"
)

tables_to_show = list(neighbor_table_path.rglob("*tsv"))
for table in tables_to_show:
    token_table = pd.read_csv(str(table), sep="\t", index_col=0)
    print(table.stem)
    display(token_table)
    print()

# ## Figure 4 - Website Walkthrough

website_visualization_path = Path("output/website_pieces")

# +
fig = plt.figure(figsize=(10, 10), dpi=60)
gs = fig.add_gridspec(nrows=1, ncols=4)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

ax1.imshow(
    plt.imread(f"{website_visualization_path}/word-lapse-trajectory.png"),
    interpolation="nearest",
)
ax2.imshow(
    plt.imread(f"{website_visualization_path}/word-lapse-frequency.png"),
    interpolation="nearest",
)
ax3.imshow(
    plt.imread(f"{website_visualization_path}/word-lapse-neighbors.png"),
    interpolation="nearest",
)

# +
panel_one = sg.fromfile(website_visualization_path / "word-lapse-trajectory.svg")
svg2png(
    bytestring=panel_one.to_str(),
    write_to=f"{website_visualization_path}/word-lapse-trajectory.png",
    dpi=600,
)
dimensions = list(
    map(lambda x: abs(int(x)), panel_one.root.attrib["viewBox"].split(" "))
)

panel_one_size = (dimensions[0] + dimensions[2], dimensions[1] + dimensions[3])

scale_x = 1
scale_y = 1

print(f"original: {panel_one_size}")
print(f"scaled:{(panel_one_size[0]*scale_x,panel_one_size[1]*scale_y)}")

panel_one = panel_one.getroot()
panel_one.scale(x=scale_x, y=scale_y)
panel_one.moveto(dimensions[0], dimensions[1])

# +
panel_two = sg.fromfile(website_visualization_path / "word-lapse-frequency.svg")

dimensions = list(
    map(lambda x: abs(int(x)), panel_two.root.attrib["viewBox"].split(" "))
)

panel_two_size = (dimensions[0] + dimensions[2], dimensions[1] + dimensions[3])
scale_x = 1
scale_y = 1

print(f"original: {panel_two_size}")
print(f"scaled:{(panel_two_size[0]*scale_x, panel_two_size[1]*scale_y)}")

panel_two = panel_two.getroot()
panel_two.scale(x=scale_x, y=scale_y)
panel_two.moveto(dimensions[0] + 723, dimensions[1])

# +
panel_three = sg.fromfile(website_visualization_path / "word-lapse-neighbors.svg")

dimensions = list(
    map(lambda x: abs(int(x)), panel_three.root.attrib["viewBox"].split(" "))
)

panel_three_size = (dimensions[0] + dimensions[2], dimensions[1] + dimensions[3])

scale_x = 1
scale_y = 1

print(f"original: {panel_three_size}")
print(f"scaled:{(panel_three_size[0]*scale_x, panel_three_size[1]*scale_y)}")

panel_three = panel_three.getroot()
panel_three.scale(x=scale_x, y=scale_y)
panel_three.moveto(dimensions[0] + 565, dimensions[1])
# -

panel_one_label = sg.TextElement(30, 30, "A", size=30, weight="bold")
panel_two_label = sg.TextElement(773, 30, "B", size=30, weight="bold")
panel_three_label = sg.TextElement(1388, 30, "C", size=30, weight="bold")

# +
figure_three = sg.SVGFigure(Unit(1900), Unit(587))

figure_three.append(
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
# display(SVG(figure_three.to_str()))
# -

# save generated SVG files
figure_three.save("output/Figure_3.svg")
svg2png(bytestring=figure_three.to_str(), write_to="output/Figure_3.png", dpi=600)

os.system(
    "convert -compress LZW -alpha remove output/Figure_3.png output/Figure_3.tiff"
)
os.system("mogrify -alpha off output/Figure_3.tiff")
