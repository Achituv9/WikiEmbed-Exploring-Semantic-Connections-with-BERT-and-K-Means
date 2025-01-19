# Clustering Wikipedia Articles Using BERT and K-Means

## Project Overview
This project aims to evaluate the effectiveness of the BERT model in capturing semantic relationships between articles. By embedding approximately 2.5 million Wikipedia pages, we used K-means clustering and the Louvain network to analyze the quality of the embeddings. Our analysis successfully identified meaningful connections between related articles, demonstrating the potential of BERT embeddings to capture semantic relationships.

For a detailed explanation of our findings, refer to the [Clustering Wikipedia Articles Using BERT and K-Means.pdf](Clustering%20Wikipedia%20Articles%20Using%20BERT%20and%20K-Means.pdf) file.

## Dataset
The dataset used in this project is available on Kaggle:  
[Plain Text Wikipedia (2020-11)](https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011)  

### Preparing the Dataset
1. Download the dataset from the Kaggle link above.
2. Extract the contents of the downloaded ZIP file.
3. Rename the extracted folder to `wikipedia_data`.

Ensure the folder structure looks like this:  
```
project_directory/
│
├── main.py
├── wikipedia_data/
│   ├── articles1.json
│   ├── articles2.json
│   ├── ...
├── most_common_categories_network.py
├── embedding_and_k_means.py
├── id_category_data.py
├── final_results.py
├── graphs.py
```

## Running the Code
1. Place the `main.py` file in the same directory as the `wikipedia_data` folder.
2. Follow these steps to execute the project:
   1. Ensure all dependencies are installed (see **Requirements** section below).
   2. Run the script using the following command:
      ```bash
      python main.py
      ```

## Requirements
You can install all required packages using the command:
```bash
pip install -r requirements.txt
```

## Outputs
The results of the analysis, including the clustering output and insights, are saved as:
- **Clustering Results**: Output files in the specified folder.
- **Detailed Findings**: [Clustering Wikipedia Articles Using BERT and K-Means.pdf](Clustering%20Wikipedia%20Articles%20Using%20BERT%20and%20K-Means.pdf).

---

This README is now clear, professional, and reader-friendly, making it easy for others to understand and reproduce your project.
