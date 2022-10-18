import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from .data import DataManager
from .models import Generator, Discriminator
from .optim import InverseSquareRootSchedule
from .options import Options
from .utils import torch_seed


class Trainer:
    def __init__(self, args: Options, dm: DataManager, generator: Generator, discriminator: Discriminator):
        self.args = args
        self.data = dm.data
        self.epoch = 0
        self.idx = 0
        self.generator = generator.cuda()
        self.discriminator = discriminator.cuda()
        self.g_optimizer = torch.optim.AdamW(
            generator.parameters(),
            args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd
        )
        self.d_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            args.lr, betas=(args.beta1, args.beta2), weight_decay=args.wd
        )
        self.g_scheduler = InverseSquareRootSchedule(self.g_optimizer, int(len(self) * args.warmup * args.epoch + 0.5))
        self.d_scheduler = InverseSquareRootSchedule(self.d_optimizer, int(len(self) * args.warmup * args.epoch + 0.5))
        self.id_to_kwdesc = dm.id_to_kwdesc
        self.collate_cuda = dm.collate_cuda

    def get_d_xy(self, g_train):
        if g_train:
            self.generator.train()
        else:
            self.generator.eval()
        x, mask, y1, y2 = self.collate_cuda(
            self.data['pytorch'][self.idx: self.idx + self.args.bs],
            self.data['keras'][self.idx: self.idx + self.args.bs]
        )
        self.idx += self.args.bs
        with torch_seed(self.args.seed, self.epoch, self.idx, 0):
            if g_train:
                z1, z2, p1, p2 = self.generator(x, mask, run_classifier=True)
            else:
                with torch.no_grad():
                    z1, z2 = self.generator(x, mask, run_classifier=False)
        dx = torch.cat([z1, z2], dim=0)
        dy = dx.new_zeros(self.args.bs * 2)
        dy[:self.args.bs] = self.args.d_smooth
        dy[self.args.bs:] = 1.0 - self.args.d_smooth
        if g_train:
            return dx, dy, p1, p2, y1, y2
        else:
            return dx, dy

    def d_step(self):
        self.discriminator.train()
        x, y = self.get_d_xy(g_train=False)
        with torch_seed(self.args.seed, self.epoch, self.idx, 1):
            p = self.discriminator(x)
        assert p.isnan().sum().item() == 0
        if self.args.do_adv:
            loss = F.binary_cross_entropy_with_logits(p, y, reduction='mean')
        else:
            loss = torch.tensor(0.0)
        stats = {
            'd_loss': loss.item(),
            'd_dacc': 1.0 - ((p >= 0.0) ^ (y >= 0.5)).float().mean().item(),
            'd_lr': self.d_scheduler.get_last_lr()
        }
        self.d_optimizer.zero_grad()
        if self.args.do_adv:
            loss.backward()
        self.d_optimizer.step()
        return stats

    def g_step(self):
        self.discriminator.eval()
        x, y, p1, p2, y1, y2 = self.get_d_xy(g_train=True)
        with torch_seed(self.args.seed, self.epoch, self.idx, 1):
            dp = self.discriminator(x)
        assert dp.isnan().sum().item() == 0
        if self.args.do_adv:
            g_loss = F.binary_cross_entropy_with_logits(dp, 1.0 - y, reduction='mean')
        else:
            g_loss = torch.tensor(0.0)
        c1_loss = F.cross_entropy(p1, y1)
        c2_loss = F.cross_entropy(p2, y2)
        stats = {
            'g_loss': g_loss.item(),
            'g_dacc': 1.0 - ((dp >= 0.0) ^ (y >= 0.5)).float().mean().item(),
            'c1_loss': c1_loss.item(),
            'c2_loss': c2_loss.item(),
            'c1_acc': (p1.argmax(dim=-1) == y1).float().mean().item(),
            'c2_acc': (p2.argmax(dim=-1) == y2).float().mean().item(),
            'g_lr': self.g_scheduler.get_last_lr()
        }
        self.g_optimizer.zero_grad()
        (g_loss + c1_loss + c2_loss).backward()
        self.g_optimizer.step()
        return stats

    def save_checkpoint(self):
        state_dict = {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            'g_scheduler': self.g_scheduler.state_dict(),
            'd_scheduler': self.d_scheduler.state_dict(),
            'id_to_kwdesc': self.id_to_kwdesc,
            'args': asdict(self.args)
        }
        os.makedirs(self.args.out_dir, exist_ok=True)
        torch.save(state_dict, os.path.join(self.args.out_dir, f"checkpoint_{self.epoch}.pt"))

    def __iter__(self):
        rng = np.random.RandomState(self.args.seed + self.epoch * 100)
        self.epoch += 1
        self.idx = 0
        rng.shuffle(self.data['pytorch'])
        rng.shuffle(self.data['keras'])
        return self

    def __next__(self):
        if self.idx + self.args.bs > len(self.data['pytorch']):
            raise StopIteration()
        d_log, g_log = [], []
        for _ in range(self.args.d_steps):
            d_log.append(self.d_step())
        g_log.append(self.g_step())
        self.d_scheduler.step()
        self.g_scheduler.step()
        return d_log, g_log

    def __len__(self):
        return len(self.data['pytorch']) // (self.args.bs * (self.args.d_steps + 1))
