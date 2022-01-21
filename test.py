import numpy as np
import pandas as pd
import os
import tqdm
import time

import torch
import torch.nn
import torch.nn.functional as F
import timm

from loader import getLoader
from config import return_args
from utils import *


def main():
    args = return_args()
    if not args.checkpoint:
        raise AttributeError('checkpoint is needed')
    main_worker(args)


def main_worker(args):
    test_loader = getLoader(args, 'test')
    
    model = timm.create_model(
        'swin_base_patch4_window7_224', pretrained=True, num_classes=2)
    checkpoint = torch.load(os.path.join('save', args.checkpoint + '.pth'))
    model.load_state_dict(checkpoint)
    model.cuda()
    model.eval()
    print('testing')
    
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    progress = ProgressMeter(
        len(test_loader),
        [batch_time],
        prefix="Testing: ")
    
    outputs = []
    index = []
    end = time.time()
    with torch.no_grad():
        for i, (img, idx) in enumerate(test_loader):
            batch_time.update(time.time() - end)

            output = model(img.cuda())
            output = F.softmax(output, dim=1)[:, 1]
            output = output.cpu().numpy().clip(min=0.005, max=0.995)
            outputs.append(output)
            index.append(idx)
            if i % args.print_freq == 0:
                progress.display_common(i)
                
    outputs = np.concatenate(outputs)
    index = np.concatenate(index)
    assert len(outputs) == len(index)
    
    df = {'id': index.tolist(), 'label': outputs.tolist()}
    df = pd.DataFrame(df)      
    df.to_csv('pred.csv', index=False)
    
    
if __name__ == '__main__':
    main()
