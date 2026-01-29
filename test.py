import yaml 
import itertools


with open("experiments_simple.yaml") as f:
    grid = yaml.safe_load(f)

keys, values = zip(*grid.items())
for combo in itertools.product(*values):
    cfg = dict(zip(keys, combo))

print (cfg)