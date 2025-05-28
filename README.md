# Benchmarking Genomic Foundation models on predicting RNA-binding protein sites.
 In this project, we have used six pre-trained models and fine-tuned them to classify RNA binding protein site prediction.
The fine-tuned scripts of all the models are in the repository. You must download the pre-trained model from the links below, and then you can use the script for your tasks.
Before downloading, please create the environment using the .txt or .yml files from the model's respective GitHub repository to install all the packages. (For GitHub repo, check the credit section.)
We have hard-coded the command-line arguments inside the fine-tuning scripts for Caduceus and RNA-FM. For the model DNABERT-1 , DNABERT-2 and GeneMask : We have used the finetuned script available in the Github repository of the authors .
## Pre-trained model links are available at below links:
DNABERT-1: https://github.com/jerryji1993/DNABERT
DNABERT-2: https://github.com/MAGICS-LAB/DNABERT_2
GeneMask: https://github.com/roysoumya/GeneMask
Caduceus: https://huggingface.co/kuleshov-group/caduceus-ps_seqlen-1k_d_model-256_n_layer-4_lr-8e-3
RNA-FM: https://github.com/ml4bio/RNA-FM?tab=readme-ov-file#rna-fm
RNAernie: https://huggingface.co/LLM-EDA/RNAErnie
We used the same script mentioned in the GitHub repository, but created our function to get the prediction output in CSV format.
## Steps:
1.	Download the pretrained model files or use the Hugging Face model directly.
2.	Create the environment and install all the necessary libraries.
3.	Convert the Fasta files to CSV (Data_preparation folder)
4.	You must convert the dataset into k-mers and save the train, validation and test files in “.tsv” format. (This is only for DNABERT-1 and GENEMASK) (Data_preparation folder).
5.	Change the Datapath and model path per your directory in the finetune script.
6.	You can use or tweak our parameters according to your requirements.
7.	Perform Inference on the predicted output of the models. (Using inference script).
## Credits
- Thanks to Yanrong Ji et al. (https://github.com/jerryji1993/DNABERT) for their work (https://www.biorxiv.org/content/10.1101/2020.09.17.301879v1).
- Thanks to Zhihan Zhou et al. (https://github.com/MAGICS-LAB/DNABERT_2) for their work (https://arxiv.org/abs/2306.15006).
- Thanks to Soumyadeep Roy et al. (https://github.com/clinical-trial/GeneMask/tree/main) for their work (https://arxiv.org/abs/2307.15933).
- Thanks to Schiff et al. (2024) (https://github.com/kuleshov-group/caduceus/tree/main) for their work (https://arxiv.org/abs/2403.03234).
- Thanks to Chen et al. (https://github.com/ml4bio/RNA-FM) for their work (https://arxiv.org/abs/2204.00300).
- Thanks to Yanrong Ji et al. (https://github.com/jerryji1993/DNABERT) for their work (https://www.biorxiv.org/content/10.1101/2020.09.17.301879v1).

