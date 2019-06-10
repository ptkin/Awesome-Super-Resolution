# Awesome Super-Resolution (Appendix)

Some supplementary materials for this repository, including building instructions, contribution guides, etc.

**Table of Contents**

- How We Build This Collection
- Difference with the Taxonomy of Our Survey
- Revision & Contribution Guide
- Contributor List



## How We Build This Collection

1. Collecting

   We used [Zotero](https://www.zotero.org/) to maintain an SR paper database, where the paper title, authors, abstract and other metadata are automatically extracted from the PDF files by Zotero and adjusted by us very roughly. The publication page, project homepage, official open source code and other related resource are collected from the Internet and entered into Zotero manually.

2. Tagging

   We manually tag these papers with some keywords like "video", "depth map", "dataset", "unsupervised", "loss function", "back projection", which is very helpful for us to write the [survey](https://arxiv.org/abs/1902.06068).

3. Exporting & Rendering

   We export the database as a CSV file, process it through Python scripts, and render it with our pre-organized document template into Markdown documents.



## Difference with the Taxonomy of Our Survey

Our survey focuses on the basic components in the image super-resolution field, so we use a method similar to "multi-label classification" to classify these papers. For example, a paper may be mentioned in the "Video Super-resolution" section, "Loss Functions" section and “Unsupervised Learning” section at the same time. 

But in this GitHub repository, we are listing various independent, complete research works and the related links, so it might be a better choice to use a method similar to "multi-class classification"  to organize these papers. So we just divide these papers into a few major categories so that readers can accurately and quickly locate the resources they need.



## Revision & Contribution Guide

Since many contents in this repository are automatically extracted or generated, there may be some errors and omissions. Although we have already checked all the resources, this situation cannot be avoided. If you find any errors or omissions with any information in this document, please contact us and we will complete the correction as soon as possible.

Due to our special collecting method, we are currently not open to handle pull requests. However, we welcome anyone to recommend new papers, project homepages, official implementations, third-party implementations that can reproduce the performance in the paper, and any other useful resources. 

All the contributors will be listed on our contributor list. We really appreciate your contributions to the community.



## Contributor List

* Zhihao WANG, [[email](mailto:ptkin@outlook.com)]

