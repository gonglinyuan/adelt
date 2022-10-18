import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from .data import DataManager
from .models import Generator, Discriminator
from .options import Options
from .trainer import Trainer


def main(args: Options):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    dm = DataManager.load(args.data_dir, args.bs)
    generator = Generator(args.dim_in, args.dim, len(dm.id_to_kwdesc['pytorch']), len(dm.id_to_kwdesc['keras']), args.drop)
    discriminator = Discriminator(args.dim, args.d_dim_hid, args.drop, args.leaky)
    trainer = Trainer(args, dm, generator, discriminator)
    d_logs, g_logs = [], []
    for _ in range(args.epoch):
        for d_log, g_log in tqdm(trainer):
            d_logs.extend(d_log)
            g_logs.extend(g_log)
        trainer.save_checkpoint()
    with open(os.path.join(args.out_dir, "log.json"), "w", encoding="utf-8") as f:
        json.dump({
            'd_logs': d_logs,
            'g_logs': g_logs
        }, f)
