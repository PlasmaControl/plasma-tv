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
