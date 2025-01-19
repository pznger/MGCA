# MGCA

***nvidia-tensorflow***         1.15.4+nv20.12

***keras***                      2.2.5

If you encounter any issues while setting up the project environment, feel free to consult me.


### path 
- data_root_path: MIND dataset

- embedding_path: 840B.300d version of the glove embedding

- KG_root_path: graph structre, pre-trained transE embeddings

  ---

  Place data into the corresponding folder and run main.jpynb to train the MGCA model.

  Dataset:

  data:Unzip MINDSmall.rar to get the processed dataset.

  KG:The detailed construction algorithm of the KG can be found in this paper https://arxiv.org/pdf/1910.11494.pdf .If you want to obtain it directly, please refer to the note section at the bottom of this document.
  
  embedding:https://nlp.stanford.edu/projects/glove/

  ---
  Here is an example of the project structure.
  
  ![1736903559573](https://github.com/user-attachments/assets/ed9dd4cc-2d64-4005-be75-6084d4eacc20)


  ---

Note: 

1.The reason the project uses NVIDIA TensorFlow is that the experimental environment is equipped with an RTX 3090 GPU, which is not compatible with the official version of TensorFlow. Therefore, if the version of TensorFlow you are using is different from mine, you may encounter issues.If you encounter any problems during the environment configuration process, you can refer to:https://blog.csdn.net/qq_73664776/article/details/135888520?spm=1001.2014.3001.5502

2.Due to potential license issues, I will not publicly release  graph structre and pre-trained transE embeddings, but I can privately share them with you. Please send an email to me to query these data.(202211070511@edu.sdust.cn).
