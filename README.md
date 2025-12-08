# decision-tree-challenge

A Kaggle competition project focused on decision tree modeling and analysis.

## Setup Instructions

```bash
source .venv/bin/activate
```
### Install Dependencies

All required packages are already installed in the virtual environment

### Start Jupyter

```bash
jupyter lab
```

## Download Kaggle Data

To download the competition data, follow these steps:

1. **Install Kaggle CLI**
   pip install kaggle
   ```

2. **Download competition data** to the `data/` folder:
   ```bash
   mkdir -p data
   kaggle competitions download -c m-2-ecap-2025 -p data/
   ```

3. **Extract the data:**
   ```bash
   cd data
   unzip "*.zip"
   cd ..
   ```
