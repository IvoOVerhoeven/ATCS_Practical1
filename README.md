<div align="center">

# ATCS Practical 1: InferSent
Learning sentence representations from natural language inference data
</div>

# Structure
```
ATCS_Practical1
├── checkpoints
│   └── model training output, including lightning/tensorboard logs and pretrained weights. Also includes all of the evaluation files. Can be downloaded at: [here](https://drive.google.com/drive/folders/1eyJRuFFR20y1e-6WXGQ-Zz7sqG02K7A5?usp=sharing).
├── data
│   ├── snli.py
│   │       code for processing and preparing SNLI using torchtext's legacy code
│   └── snli_hugingface.py
│           code for processing and preparing SNLI using Huggingface's dataset package
├── evaluation
│   │     code needed for evaluating models
│   ├── generate_embeddings.py
│   │       creates embeddings for all the premises in SNLI. Not actually used (files are too big)
│   ├── importance_weights.py
│   │       generates weights for individual words in a sentence. Includes the maxpool propensity score for InferSent
│   ├── retrieval.py
│   │       some functions for finding similar sentences in the embeddings. Again, not actually used
│   └── visualization.py
│           plenty of modules for pretty visuals in the Analysis notebook
├── models
│   │     code that combines modules into coherent structure (PyTorch-Lightning)
│   └── InferSent.py
│         sentence encoder. Allows for forward pass on SNLI, or encode method for general purpose sentence encodings
├── modules
│   │     individual PyTorch modules for sentence encoders
│   ├── classifier.py
│   │       MLP classifier
│   ├── embedding.py
│   │     lookup embedding with GloVe-8B-300D vectors
│   └── encoder.py
│         all the actual encoders
├── tensorboard_imgs
│   └──   some pngs in case Tensorboard won't load inside ipynb
├── utils
│   ├── reproducibility.py
│   │       some methods for reproducibility's sake
│   ├── text_processing.py
│   │       modules for taking text to proper input and back
│   └── timing.py
│           times the training runs
├── Analysis.ipynb
│   └── Jupyter notebook with the analysis and summary report
├── InferSent_train.py
│   └── script for training
├── InferSent_SentEval.py
│   └── script for evaluating on SentEval, with certain profiles
├── snli_embeddings.py
│   └── script for generating sentence embeddings on SNLI sentences
├── snli_evaluate.py
│   └── script for evaluating on SNLI only
└── snli_linguistic.py
    └── file for producing interesting analyses, using Spacy's tokenizer and finding misclassified sentences
```
# Environment
See 'acts_environment_lisa.yml' for training environment.
Primary packages used are:
* pytorch=1.8.0
* torchtext=0.9.0
* pytorch-lightning=1.2.5
* spacy=3.0.5
* scikit-learn=0.24.1 (SentEval dependency)
# Running Code
## SNLI Training
Script for training various sentence encoders on SNLI.
```
python InferSent_train.py
```
To retrain evaluated models, command line options are (version 3 already exists):
```
--encoder Baseline --bidirectional False --debug False --version 4
--encoder Simple --bidirectional False --hidden_dims 4096 --debug False --version 4
--encoder Simple --bidirectional True --hidden_dims 4096 --debug False --version 4
--encoder MaxPool --bidirectional True --hidden_dims 4096 --debug False --version 4
```
## SNLI Evaluation
Script for evaluating models on the SNLI test and validation subsets.
```
python snli_evaluate.py
```

Option 'encoder' specifies which architecture to load in ('Baseline', 'Simple', 'BiSimple', 'BiMaxPool')
Option 'version' specifies which version to load in. Should correspond to version number in ./checkpoints

Will create pickle file in the corresponding ./checkpoints directory
## SentEval Evaluation
Script for evaluating on the SentEval tasks.
```
python InferSent_SentEval.py
```
To evaluate on the same profiles, use the following command line options:
```
--encoder BiMaxPool --tasks InferSent --config default --senteval_path $HOME/SentEval
--encoder Baseline --tasks InferSent --config default --senteval_path $HOME/SentEval
--encoder Simple --tasks InferSent --config default --senteval_path $HOME/SentEval
--encoder BiSimple --tasks InferSent --config default --senteval_path $HOME/SentEval

--encoder BiMaxPool --tasks probing_all --config default --senteval_path $HOME/SentEval
--encoder Baseline --tasks probing_all --config default --senteval_path $HOME/SentEval
--encoder Simple --tasks probing_all --config default --senteval_path $HOME/SentEval
--encoder BiSimple --tasks probing_all --config default --senteval_path $HOME/SentEval
```

Option 'tasks' specifies which task profile. Options are 'all', 'infersent', 'working', 'transfer_all' or 'probing_all'.
Option 'config' specifies which encoder configuration. Options are 'default' and 'fast'. See the SentEval git repo for details.
Option 'version' specifies which version to load in. Should correspond to version number in ./checkpoints

Will create pickle file in the corresponding ./checkpoints directory
## Analysis files
Script for returning some interesting analysis files.
```
python snli_linguistic.py
```

Option 'encoder' specifies which architecture to load in ('Baseline', 'Simple', 'BiSimple', 'BiMaxPool')
Option 'version' specifies which version to load in. Should correspond to version number in ./checkpoints

Will create pickle file in the corresponding ./checkpoints directory