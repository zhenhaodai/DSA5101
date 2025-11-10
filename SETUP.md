# Setup Instructions

This guide will help you set up the environment and download the MovieLens 20M dataset for the DSA5101 assignment.

## Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Internet connection for downloading dataset (~190MB)

## Step 1: Install Python Dependencies

Open a terminal/command prompt in the project directory and run:

```bash
pip install -r requirements.txt
```

This will install all required packages:
- numpy
- pandas
- scikit-learn
- matplotlib
- seaborn
- networkx
- scipy
- tqdm

## Step 2: Download MovieLens 20M Dataset

Choose the appropriate script for your operating system:

### For Linux/Mac:

```bash
bash download_data.sh
```

Or make it executable first:

```bash
chmod +x download_data.sh
./download_data.sh
```

### For Windows (PowerShell):

```powershell
.\download_data.ps1
```

**Note:** You may need to enable script execution in PowerShell:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Manual Download (if scripts fail):

1. Visit: https://grouplens.org/datasets/movielens/20m/
2. Download `ml-20m.zip`
3. Extract to `./data/` directory
4. Ensure the structure is: `./data/ml-20m/ratings.csv` and `./data/ml-20m/movies.csv`

## Step 3: Verify Setup

After completing the above steps, your directory structure should look like:

```
DSA5101/
├── data/
│   └── ml-20m/
│       ├── ratings.csv (~500MB)
│       └── movies.csv (~1MB)
├── models/              (will be created by notebook)
├── figures/             (will be created by notebook)
├── DSA5101_MovieLens_Analysis.ipynb
├── requirements.txt
├── download_data.sh
├── download_data.ps1
└── SETUP.md
```

## Step 4: Run the Notebook

1. Start Jupyter Notebook or JupyterLab:
   ```bash
   jupyter notebook
   ```

2. Open `DSA5101_MovieLens_Analysis.ipynb`

3. Run all cells (Cell → Run All)

## Expected Runtime

- **Full Training Mode** (`TRAIN_MODELS = True`): 30-60 minutes
  - SVD: ~10 seconds
  - ALS: ~2 minutes
  - ItemKNN: ~15 seconds
  - PageRank: ~10-30 minutes (slowest)
  - Hybrid: ~5 seconds

- **Load Model Mode** (`TRAIN_MODELS = False`): ~2 minutes
  - Requires pre-trained models in `./models/` directory

## Troubleshooting

### "wget: command not found" (Linux/Mac)

Install wget:
- **Ubuntu/Debian**: `sudo apt-get install wget`
- **Mac**: `brew install wget`

Or use curl instead:
```bash
curl -O http://files.grouplens.org/datasets/movielens/ml-20m.zip
```

### "unzip: command not found" (Linux)

Install unzip:
```bash
sudo apt-get install unzip
```

### PowerShell Execution Policy Error (Windows)

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Slow Download

If download is slow, you can:
1. Use a download manager to download the zip file manually
2. Download from a mirror if available
3. Use a VPN if access is restricted in your region

## Dataset Information

- **Name:** MovieLens 20M
- **Source:** GroupLens Research @ University of Minnesota
- **Size:** ~190MB compressed, ~500MB extracted
- **Ratings:** 20 million ratings
- **Users:** 138,000 users
- **Movies:** 27,000 movies
- **Rating Scale:** 0.5 to 5.0 stars
- **License:** Public dataset for research and education

## Additional Help

If you encounter any issues:

1. Check that Python and pip are properly installed:
   ```bash
   python --version
   pip --version
   ```

2. Verify all packages are installed:
   ```bash
   pip list
   ```

3. Check disk space (need ~1GB free for dataset and models)

4. Ensure internet connection is stable during download

For more information, visit:
- MovieLens: https://grouplens.org/datasets/movielens/
- Project documentation: (add your project URL here)
