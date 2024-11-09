# plasma-tv
Work on plasma tv image data

This repo is set up to work with both the raw data and any synthetic data. It was a bit unorganized so I reorganized it, but this might cause some directory issues.

Directories of interest:
1. notebooks (intended workflows)
1. notebooks_temp (depracated, will be removed)
1. cam_geo (used for generating synthetic data using camera geometry)
1. eq_field (intended to get x-value and strike point value from magnetic field)

# Creating a Virtual Environment
```anaconda (recommended on Princeton HPC)
module load anaconda3/2024.6
conda env create -f environment.yml
```
```python
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
pip install -e .
```

This way, you can go into src and modify the code, and you don't have to reinstall. And also then you can directly import like `from src.idk import chicken as ch`

# Copying Videos From Aza

scp -r -o 'ProxyCommand ssh -p 2039 chenn@cybele.gat.com -W %h:%p' chenn@omega.gat.com:/cscratch/chenn/12_03_2024.h5 /scratch/gpfs/nc1514/plasma-tv/data/external/toksearch

rsync -a --ignore-existing -P -e 'ssh -o "ProxyCommand ssh -p 2039 chenn@cybele.gat.com -W %h:%p"' chenn@omega.gat.com:/cscratch/chenn/tangtv/*.pkl /scratch/gpfs/nc1514/plasma-tv/data/raw/12_03_2024

Iamahamburger123!!!Yaaa