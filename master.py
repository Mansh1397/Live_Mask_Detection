from live_mask_detection import detect
from training import train
import sys

task = sys.argv[1]

if task == 'train':
    train()
    detect()
elif task == 'detect':
    detect()
else:
    print('Incorrect Task entered. Please enter "train" or "detect".')
