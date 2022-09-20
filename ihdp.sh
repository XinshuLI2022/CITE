mkdir results
mkdir results/ihdp

mkdir propensity
mkdir propensity/ihdp

python cite_param_search.py configs/ihdp.txt 1

python evaluate.py configs/ihdp.txt 1




