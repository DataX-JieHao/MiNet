# MiNet
### MiNet is a gene- and pathway-based deep neural netowrk for multi-omics data intergration to predict cancer survival outcomes.
# Get Started
## Example Datasets
To get started, you need to download example datasets from URLs as below:

[Train data](http://datax.kennesaw.edu/MiNet/gbm_std_imputed_train_15.csv) 

[Validation data](http://datax.kennesaw.edu/MiNet/gbm_std_imputed_valid_15.csv)

[Test data](http://datax.kennesaw.edu/MiNet/gbm_std_imputed_test_15.csv)

[Gene Sparse Mask](http://datax.kennesaw.edu/MiNet/gbm_binary_gene_mask.npz)

[Pathway Sparse Mask](http://datax.kennesaw.edu/MiNet/gbm_binary_pathway_mask.npz)

## Training, Validation and Evaluation of MiNet
Run.py: to train the model with the inputs from train.csv. Hyperparmeters are optimized by grid search automatically with validation.csv. C-index is used to evaluate the model performance with test.csv.

