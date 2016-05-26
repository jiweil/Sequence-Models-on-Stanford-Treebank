# Sequence models in paper "When Are Tree Structures Necessary for Deep Learning of Representations"

Implementations of two sequence models (bi-direcitonal LSTM sequence model and Hierarchical sequence model) described in Section Section 3.1 on Stanford Treebank dataset

## Requirements:
GPU is favaorable. The code supports both GPU and CPU. Estimated running time on a GPU machine (K40) is 15-20 minutes and 10-14 hours on a CPU machine.

matlab>=2014

memory>=4GB

## Folders

Bi_LSTM: Bidirectional LSTM sequence models. 
Root File: BiLSTM.m.  file "lr001" contains store results in 10 different tries. You can get fine-grained accruacy around 0.50. 

Hier_Bi_LSTM : Hierarhical (two-level, word level and clause level) LSTM sequence models. 
Root file: HierLSTM.m

## DownLoad [Data](http://cs.stanford.edu/~bdlijiwei/data_sentiment.tar)
-`sentiment_glove_300.txt`: 300 dimentional pre_trained embeddings from Glove.

-`sequence_train.txt`: training file for Bi_LSTM. First token specifies sentiment label. Each line correspond to a parse tree constituent from the original dataset. Lines are ordered by number of its containing tokens. 

-`sequence_dev_root.txt`: development file for Bi_LSTM.

-`sequence_test_root.txt`: testing file for Bi_LSTM.

-`sequence_train_segment.txt`: training file for Hier_Bi_LSTM with original sequences being broken into a chunk of clause-like units upon punctuations (i.e., period, comma and question mark). First token in the first line of each chunk specifies current sentiment label. Chunks are orderd by number of its containing clauses.

-`sequence_dev_root_segment.txt`: development file for Hier_Bi_LSTM.

-`sequence_test_root_segment.txt`: testing file for Hier_Bi_LSTM.

For any pertinent questions regarding the code (e.g., bugs etc), feel free to email jiweil@stanford.edu

```latex
@article{li2015tree,
    title={When Are Tree Structures Necessary for Deep Learning of Representations?},
    author={Li, Jiwei and Luong, Minh-Thang and Jurafsky, Dan and Hovy, Eudard},
    journal={arXiv preprint arXiv:1503.00185},
    year={2015}
}
```
