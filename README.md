# Nutrition_NLP

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/github/license/sia-jaf/Nutrition_NLP)

This repository contains code and notebooks for deep learning and NLP analysis on nutrition and food description, focusing on clustering, classification, and embedding of food descriptions and nutrients.

## Goal

This project applies Natural Language Processing (NLP) to structure and analyze complex, unstructured food description data from the USDA. The primary objective was to create parsimonious, sensible categories for the foods in the database. Applying natural language processing and k-means clustering to unstructured text enables downstream applications such as classifying new, previously unseen foods.


## Project Workflow

1. **Data Preparation**    
	- Loads and preprocesses food and nutrient data from the USDA FoodData Central datasets.  
	- Cleans, merges, and prepares tables for downstream analysis.
    - See notebook [`FDCnlp1prepare.ipynb`](notebooks/FDCnlp1prepare.ipynb).

2. **Sentence Embeddings**  
	- Applies sentence transformers to food descriptions to generate embeddings.  
	- Outputs a large dataframe with food items and their vector representations.
    - See notebook [`FDCnlp2sentencetransformers.ipynb`](notebooks/FDCnlp2sentencetransformers.ipynb).

3. **Clustering**   
	- Performs KMeans clustering on the sentence embeddings and nutrient profiles.  
	- Assigns cluster labels to each food item for both embedding-based and nutrient-based clusters.
    - See notebook [`FDCnlp3kmeanscluster.ipynb`](notebooks/FDCnlp3kmeanscluster.ipynb).

4. **Cluster Analysis**   
	- Analyzes the nutrient composition of each cluster.  
	- Compares clusters to USDA-supplied food categories.
    - See notebook [`FDCnlp4clusternutrients.ipynb`](notebooks/FDCnlp4clusternutrients.ipynb)

5. **Cluster Evaluation**  
	- Evaluates the quality of clusters using silhouette scores and gap statistics.  
	- Assigns descriptive names to clusters and compares cluster assignments to known food categories.  
	- Finds that embedding-based clusters outperform nutrient-based clusters in terms of interpretability.
    - See notebook [FDCnlp5evaluateclusters.ipynb](notebooks/FDCnlp5evaluateclusters.ipynb).

## Notes

- **Data**: Due to sheer size of the raw data sizes provided by USDA, we recommend downloading the datasets from [DataCenter](https://fdc.nal.usda.gov/download-datasets). Then starting at the beginning of the workflow.
- **Path**: The path to the data on notebooks points to the google drive. For local implementation, that path needs to be adjusted. 
- **Computational resource**: This project was mainly run on Google Colab engine with a GPU. For computational 

## Usage
- Download the raw data from USDA.
- Run the notebooks in order for a full pipeline from data preparation to cluster evaluation.
- Each notebook is self-contained and documents its own dependencies and outputs.

## Requirements

- For creating a Python env see `requirements.txt`

## Update

Authors will turn the notebooks into a python library to provide more robust access to the code. 

