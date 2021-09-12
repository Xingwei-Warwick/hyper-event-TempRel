Poincaré Event Temporal Embeddings and Hyperbolic GRU for Event TempRel Extraction
-----
-----

This is the repository of EMNLP 2021 paper "Extracting Event Temporal Relations via Hyperbolic Geometry".

The code is modified based on the open-source repository of [1], which is one of the compared methods in our paper.

>[1] Qiang Ning, Sanjay Subramanian, and Dan Roth. 2019. An improved neural baseline for temporal relation extraction. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 6204–6210.


If our code helps, please consider adding the following reference:
>Xingwei Tan, Gabriele Pergola, and Yulan He. 2021. Extracting Event Temporal Relations via Hyperbolic Geometry. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP).

Reproduction
-----

The data files of MATRES and TCR are already inclued in `data/` folder.

Make sure you have also downloaded the `ser/` folder, which contains ELMo and Siamese pre-trained models:

    wget https://cogcomp-public-data.s3.amazonaws.com/processed/NeuralTemporal-EMNLP19-ser/NeuralTemporal-EMNLP19-ser.tgz
    tar xzvf NeuralTemporal-EMNLP19-ser.tgz

Run the following script for HGRU with RoBERTa fine-tunning:

    ./reproduce_HGRU_roberta.sh

Run the followings for Poincare Event embeddings with RoBERTa fine-tunning:

    ./reproduce_myPoincare_roberta.sh

We are not able to include the state_dict of the models in this repository, because of the file size. Thus, the above two scripts include training. Please make sure you have modified the batch size based on your device.

The following two are models are with static RoBERTa. We included the state_dict of the models in the `models/` folder.

Run the following script for HGRU with static RoBERTa:

    ./reproduce_HGRU_static_roberta.sh

Run the followings for Poincare Event embeddings with static RoBERTa:

    ./reproduce_myPoincare_static_roberta.sh

Dependancy
------

    allennlp                  0.9.0       
    cudatoolkit               10.0.130
    matplotlib                3.1.1   
    nltk                      3.4.5   
    numpy                     1.17.2  
    numpy-base                1.17.2  
    python                    3.8.3   
    pytorch                   1.6.0   
    transformers              3.4.0
    scikit-learn              0.21.3  
    scipy                     1.3.1   
    spacy                     2.1.8   
    tqdm                      4.36.1
    geoopt                    0.3.1