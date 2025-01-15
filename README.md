# MGCA
***nvidia-tensorflow***         1.15.4+nv20.12
***keras***                      2.2.5

### path 
- data_root_path: MIND dataset

- embedding_path: 840B.300d version of the glove embedding

- KG_root_path: graph structre, pre-trained transE embeddings

  *** Python files***

  ---

  main.jpynb 、Place the data into the corresponding folder and run main.jpynb to train the MGCA model.

  Dataset: data:https://msnews.github.io/#getting-start
  KG:https://drive.google.com/file/d/1mJuB9OEDY7CNSz8Lfko1IrnuxHJoPu-L/view?usp=sharing
  embedding:https://nlp.stanford.edu/projects/glove/

  ---
  Here is an example of the project structure.
![1736903559573](https://github.com/user-attachments/assets/ed9dd4cc-2d64-4005-be75-6084d4eacc20)

  ---

  ---

  ---

Note: The example in the case study is not a impression obtained from the preprocessed test set, so you cannot find this impression in the tsv file. Instead, they are constructed by sampling news and entity relationships from the dataset. This approach allows us to obtain the most typical examples to illustrate the applicability of the model. 


  

  
