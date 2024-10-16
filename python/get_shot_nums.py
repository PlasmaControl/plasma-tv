from pathlib import Path
input_dir = 'aza'
input_path = Path('../data/raw/tv_images') / input_dir
files = list(input_path.rglob('*.pkl'))

shot_nums = []
for file in files:
    shot_nums.append(int(file.stem.split('_')[-1]))
print(shot_nums)