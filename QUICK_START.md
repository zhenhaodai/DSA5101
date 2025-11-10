# Quick Start Guide

Get started with the DSA5101 MovieLens Analysis in 3 simple steps:

## Step 1: Install Dependencies (30 seconds)

```bash
pip install -r requirements.txt
```

## Step 2: Download Dataset (~5 minutes)

**Linux/Mac:**
```bash
bash download_data.sh
```

**Windows PowerShell:**
```powershell
.\download_data.ps1
```

## Step 3: Run Notebook

```bash
jupyter notebook DSA5101_MovieLens_Analysis.ipynb
```

Then click **Cell → Run All**

---

## Expected Results

- **With Training** (`TRAIN_MODELS = True`): ~30-60 minutes
- **With Pre-trained Models** (`TRAIN_MODELS = False`): ~2 minutes

## Need Help?

See [SETUP.md](SETUP.md) for detailed instructions and troubleshooting.

---

## File Structure

```
DSA5101/
├── requirements.txt              ← Install this first
├── download_data.sh             ← Run this second (Mac/Linux)
├── download_data.ps1            ← Run this second (Windows)
├── DSA5101_MovieLens_Analysis.ipynb  ← Run this third
├── SETUP.md                     ← Detailed instructions
└── QUICK_START.md              ← You are here!
```

That's it! 🚀
