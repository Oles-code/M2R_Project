# M2R_Project
Learning Causal Structure and Effects from Observational Data M2R project for Imperial College London


### Setup:
1. Create a venv in python 3.12 (need older version to use the dataset)
    run this:
    python3.12 -m venv m2r_venv                         
    source m2r_venv/bin/activate      
2. Required packages:
    Saved in requirements.txt, to install run: 
        pip install -r requirements.txt

    if this fails then run:
        pip install setuptools
        pip install causalbench --use-deprecated=legacy-resolver
    to download the dataset + dependencies 
