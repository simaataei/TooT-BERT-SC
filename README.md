# TooT-BERT-SC

TooT-BERT-SC is a BERT based classification model, predicting eleven substrate classes of transmembrane transport proteins. The list of classes is as follows.

1. Nonselective
2. Water
3. Inorganic cations
4. Inorganic anions
5. Organic anions
6. Organo-oxygens
7. Amino acids and derivatives
8. Other organonitrogens
9. Nucleotides
10. Organic heterocyclics
11. Miscellaneous

This model is based on Prot-BERT-BFD model fine tuned on Substrate Class (SC) dataset. The BERT model is followed by a linear layer for classfication using softmax function.

#Usage:

The program could be run using the following command:


```console
python run.py [input_fasta_file] [output_file]
```

For example:

```console
python run.py Datasets/test.fasta out.txt
```
The file "test.fasta" is the input file containing protein sequences in fasta format and "out.txt" contains the id of the test sequence followed by the prediction.
