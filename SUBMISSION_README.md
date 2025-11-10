# DSA5101 Assignment Submission Guide

## Files for Submission

### Required Files:
1. **DSA5101_MovieLens_Analysis.ipynb** - Main notebook (self-contained)
2. **DSA5101_MovieLens_Analysis.html** - HTML export
3. **figures/** - Generated visualizations (optional, included in HTML)
4. **models/** - Trained models (optional, for reproducibility)

---

## How to Generate HTML

### Option 1: Using Jupyter Notebook
```bash
jupyter nbconvert --to html DSA5101_MovieLens_Analysis.ipynb
```

### Option 2: Using Jupyter Lab
1. Open `DSA5101_MovieLens_Analysis.ipynb` in Jupyter Lab
2. Click **File → Save and Export Notebook As → HTML**
3. Save as `DSA5101_MovieLens_Analysis.html`

### Option 3: Using nbconvert command line
```bash
python -m nbconvert --to html DSA5101_MovieLens_Analysis.ipynb
```

---

## Running the Notebook

### Prerequisites:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn networkx tqdm jupyter
```

### Data Download:
```bash
# Download MovieLens 20M
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip -d data/
```

### Execution Options:

#### Full Training (Recommended for first run):
```python
TRAIN_MODELS = True  # In cell 17
```
**Time:** ~30-60 minutes (PageRank is slowest)

#### Quick Run (Using cached results):
```python
TRAIN_MODELS = False  # In cell 17
```
**Time:** ~1-2 minutes

---

## Notebook Structure

1. **Title & Student Info** - Add your name and ID
2. **Introduction** - Motivation and objectives
3. **Dataset** - MovieLens 20M description
4. **Methodology** - Algorithm descriptions
5. **Implementation** - Complete source code (self-contained)
6. **Results** - Performance comparison tables and charts
7. **Discussion** - Analysis and insights
8. **Conclusion** - Key findings
9. **References** - Academic citations

---

## Key Features

### Self-Contained Code
All source code is embedded in the notebook:
- Data Loader (Cell 6)
- Evaluation Metrics (Cell 8)
- Recommender Systems (Cell 10)
- Clustering (Cell 12)
- Visualization (Cell 14)

### Model Saving
Trained models are saved to `./models/` directory for reusability:
- svd_recommender.pkl
- als_recommender.pkl
- itemknn_recommender.pkl
- pagerank_recommender.pkl
- hybrid_recommender.pkl

### Progress Tracking
- All long operations show progress bars (tqdm)
- Training can be skipped using cached results
- Estimated times provided for each model

---

## Algorithms Implemented

### Recommendation Systems:
1. **SVD** - Traditional matrix factorization
2. **ALS** - Alternating Least Squares (modern MF)
3. **ItemKNN** - Item-based collaborative filtering
4. **PageRank** - Graph-based recommendation
5. **Hybrid** - Combines SVD + PageRank

### Clustering:
1. **K-means** - Partition-based clustering
2. **Hierarchical** - Agglomerative clustering with Ward linkage
3. **PCA** - Dimensionality reduction

### Evaluation Metrics:
- **Rating Prediction:** MAE, RMSE
- **Ranking Quality:** Precision@K, Recall@K, NDCG@K, Hit Rate
- **Clustering:** Silhouette, Davies-Bouldin, Calinski-Harabasz

---

## Expected Results

### Rating Prediction (MAE):
- **Best:** ALS (0.90)
- ItemKNN (0.85)
- PageRank (1.85)
- Hybrid (2.15)
- SVD (2.60)

### Ranking Quality (NDCG@10):
- **Best:** ItemKNN (0.30)
- ALS (0.28)
- PageRank (0.22)
- Hybrid (0.19)
- SVD (0.15)

---

## Troubleshooting

### Issue: PageRank Training Too Slow
**Solution:** Set `TRAIN_MODELS = False` and use cached results

### Issue: Memory Error
**Solution:** Reduce sample size in Data Loading cell:
```python
loader.load_data(sample_size=1000000)  # Use 1M instead of 2M
```

### Issue: Data Not Found
**Solution:** Download MovieLens 20M:
```bash
mkdir -p data
cd data
wget http://files.grouplens.org/datasets/movielens/ml-20m.zip
unzip ml-20m.zip
cd ..
```

---

## Academic Integrity

This implementation was developed with assistance from Large Language Models (Claude) for:
- Code structure and optimization
- Algorithm implementation
- Documentation and comments

All code has been manually reviewed and tested. The analysis, insights, and conclusions are original work.

---

## Contact

For questions about this assignment, contact:
- **Student:** [Your Name]
- **Email:** [Your Email]
- **Student ID:** [Your ID]

---

## License

This is an academic assignment for DSA5101. All data from:
- GroupLens Research: https://grouplens.org/datasets/movielens/

Code uses standard libraries:
- NumPy, Pandas, Scikit-learn, NetworkX, Matplotlib
