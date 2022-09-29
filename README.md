# CITE

## Introduction
This is the official implementation for paper ["Contrastive Individual Treatment Effects Estimation" ]()  

The code is built on the [CFR](https://arxiv.org/abs/1606.03976), [ABCEI](https://arxiv.org/abs/1904.13335), [SITE](https://proceedings.neurips.cc/paper/2018/hash/a50abba8132a77191791390c3eb19fe7-Abstract.html) for fair comparison.  

## Environments

Same with the implementation of [CounterFactual Regression (CFR)](https://github.com/clinicalml/cfrnet).  

## Usage

**data:** we give ihdp dataset as an example here. You can download other datasets from [here](https://www.fredjo.com/)  
**training:** python cite_param_search.py <config_file> <num_runs>  
**evaluating:** python evaluate.py <config_file> <num_runs>  
**simple command example:** sh ihdp.sh

## Citation

Please consider citing this paper if it is helpful for you.












