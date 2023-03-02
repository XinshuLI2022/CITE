# CITE

## Introduction
This is the official implementation for paper ["Contrastive Individual Treatment Effects Estimation" ]()  

The code is built on the [CFR](https://arxiv.org/abs/1606.03976), [ABCEI](https://arxiv.org/abs/1904.13335), [SITE](https://proceedings.neurips.cc/paper/2018/hash/a50abba8132a77191791390c3eb19fe7-Abstract.html) for fair comparison.  

## Environments

Same with [the implementation of CounterFactual Regression (CFR)](https://github.com/clinicalml/cfrnet).  

## Usage

**data:** we give ihdp dataset as an example here. You can download other datasets from [here](https://www.fredjo.com/)  
**training:** python cite_param_search.py <config_file> <num_runs>  
**evaluating:** python evaluate.py <config_file> <num_runs>  
**simple command example:** sh ihdp.sh

## Citation

Please consider citing this paper if it is helpful for you.

@inproceedings{DBLP:conf/icdm/Li022,
  author    = {Xinshu Li and
               Lina Yao},
  editor    = {Xingquan Zhu and
               Sanjay Ranka and
               My T. Thai and
               Takashi Washio and
               Xindong Wu},
  title     = {Contrastive Individual Treatment Effects Estimation},
  booktitle = {{IEEE} International Conference on Data Mining, {ICDM} 2022, Orlando,
               FL, USA, November 28 - Dec. 1, 2022},
  pages     = {1053--1058},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/ICDM54844.2022.00130},
  doi       = {10.1109/ICDM54844.2022.00130},
  timestamp = {Thu, 02 Feb 2023 14:29:00 +0100},
  biburl    = {https://dblp.org/rec/conf/icdm/Li022.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}












