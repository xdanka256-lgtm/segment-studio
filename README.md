# Segment Studio 
Segment Studio is a Streamlit application for automatic customer segmentation using K-Means clustering. 
## Features
- Upload a CSV file 
- Display the dataset 
- Normalize numeric features using StandardScaler
- Calculate WCSS for multiple K values
- Display an Elbow Plot 
- Calculate Silhouette Score 
- Automatically recommend the best K using Silhouette Score 
- Create clusters using K-Means
- Show cluster sizes
- Calculate numeric averages for each cluster
- Find the most common categorical values 
- Use a local Llama model through Ollama to generate: 
- Cluster name 
- Short cluster description 
- Export the final dataset with a `name_cluster` column 
## Technologies 
- Python 
- Streamlit 
- Pandas 
- Scikit-learn 
- Matplotlib
- Ollama
- Llama 3.2 
## How to Run 
Install the required libraries: 
```bash
pip install streamlit pandas scikit-learn matplotlib requests
