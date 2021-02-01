import csv
from pathlib import Path

file = Path.cwd().joinpath('spiral_validation.txt')
newFile = Path.cwd().joinpath('spiral_validation0.txt')
new = []

with open(file) as f:
    reader = csv.reader(f)
    for row in reader:
        old = int(row[0])
        new.append(old*4)
        new.append(old*4+1)
        new.append(old*4+2)
        new.append(old*4+3)

with open(newFile, 'w') as f:
    writer = csv.writer(f)
    for el in new:
        writer.writerow([el])
