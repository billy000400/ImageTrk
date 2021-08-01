'''
This script is to plot the learning curve from a .csv file
'''

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys

cwd = Path.cwd()
files = sorted(Path('.').glob('*.csv'))
fileNames = [file.name for file in files]
print("[INFO] Please type in a record name among below:")
for n in fileNames:
    print(n)
ans = input('-->')

if cwd.joinpath(ans).exists():
    df = pd.read_csv(cwd.joinpath(ans), index_col=None)
elif cwd.joinpath(ans+'.csv').exists():
    df = pd.read_csv(cwd.joinpath(ans+'.csv'), index_col=None)
else:
    print(f'[ERROR] Your input "{str(cwd.joinpath(ans))}" is not a valid record' )
    sys.exit()

epoch = df['epoch']
loss = df['loss']
acc = df.iloc[:,2]
val_loss = df['val_loss']
val_acc = df.iloc[:,-1]

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))

ax1.plot(epoch, loss, label='train loss')
ax1.plot(epoch, val_loss, label='val loss')
ax1.set(xlabel='epoch', ylabel='loss', title='Loss vs. Epoch Plot')
ax1.legend()

ax2.plot(epoch, acc, label='train acc')
ax2.plot(epoch, val_acc, label='val acc')
ax2.set(xlabel='epoch', ylabel='accuracy (%)', title='Accuracy vs. Epoch Plot')
ax2.legend()

plt.show()
plt.close()
