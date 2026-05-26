# M2R Project
**Learning Causal Structure and Effects from Observational Data**

M2R project for Imperial College London

---

## Setup

### 1. Create a Virtual Environment
This project requires Python 3.12.

```bash
python3.12 -m venv m2r_venv
source m2r_venv/bin/activate
```

### 2. Install Dependencies

Run the following to install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If this fails, install manually:

```bash
pip install setuptools
pip install causalbench --use-deprecated=legacy-resolver
pip install lingam
```

This installs the dataset and its dependencies, along with the LiNGAM model library.