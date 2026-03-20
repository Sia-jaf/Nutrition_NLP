# Nutrition_NLP

![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/github/license/your-username/Nutrition_NLP)

This repository contains code and notebooks for deep learning and NLP analysis on nutrition and food description, focusing on clustering, classification, and embedding of food descriptions and nutrients.

This project was mainly run on Google Colab engine with a GPU. For computational 

## Project Workflow

1. **Data Preparation**  
	- Notebook: `FDCnlp1prepare.ipynb`  
	- Loads and preprocesses food and nutrient data from the USDA FoodData Central datasets.  
	- Cleans, merges, and prepares tables for downstream analysis.

2. **Sentence Embeddings**  
	- Notebook: `FDCnlp2sentencetransformers.ipynb`  
	- Applies sentence transformers to food descriptions to generate embeddings.  
	- Outputs a large dataframe with food items and their vector representations.

3. **Clustering**  
	- Notebook: `FDCnlp3kmeanscluster.ipynb`  
	- Performs KMeans clustering on the sentence embeddings and nutrient profiles.  
	- Assigns cluster labels to each food item for both embedding-based and nutrient-based clusters.

4. **Cluster Analysis**  
	- Notebook: `FDCnlp4clusternutrients.ipynb`  
	- Analyzes the nutrient composition of each cluster.  
	- Compares clusters to USDA-supplied food categories.

5. **Cluster Evaluation**  
	- Notebook: `FDCnlp5evaluateclusters.ipynb`  
	- Evaluates the quality of clusters using silhouette scores and gap statistics.  
	- Assigns descriptive names to clusters and compares cluster assignments to known food categories.  
	- Finds that embedding-based clusters outperform nutrient-based clusters in terms of interpretability.

## Data

- The `data/` folder contains all raw and processed data files, including:
  - USDA FoodData Central CSVs
  - Parquet files with processed and clustered data
- The path to the data on notebooks points to the google drive. For local implementation, that path needs to be adjusted. 

## Usage

- Run the notebooks in order for a full pipeline from data preparation to cluster evaluation.
- Each notebook is self-contained and documents its own dependencies and outputs.

## Requirements

- For creating a Python env see `requirements.txt`
- 


