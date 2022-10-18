import os

import numpy as np
import pandas as pd

from adelt.dictionary_generate import generate_greedy_rank
from adelt.evaluator import Evaluator
from adelt.options import Options
from adelt.train import main as adv_train
from adelt.translate import translate_all


def _evaluate(bpe_path, vocab_path, adv_root, checkpoint, eval_data_keyword_path, eval_data_translate_path):
    evaluator = Evaluator.load(
        bpe_path,
        vocab_path,
        adv_root,
        checkpoint,
        eval_data_keyword_path,
        eval_data_translate_path
    )
    keyword_match, p_rank, k_rank = generate_greedy_rank(evaluator, "cos")
    p_out, k_out = translate_all(evaluator, keyword_match)
    m = {
        'keyword_unsupervised': evaluator.evaluate_keyword_unsupervised(keyword_match, 100),
        'keyword_unconstrained': evaluator.evaluate_keyword_unconstrained(),
        'keyword_constrained': evaluator.evaluate_keyword_constrained(keyword_match, p_rank, k_rank, quiet=True),
        'translate_pytorch': evaluator.evaluate_translate(*p_out, "pytorch", quiet=True),
        'translate_keras': evaluator.evaluate_translate(*k_out, "keras", quiet=True)
    }
    return [
        m['keyword_constrained']['pytorch-keras.all.hit1'] * 100,
        m['keyword_constrained']['keras-pytorch.all.hit1'] * 100,
        m['keyword_constrained']['pytorch-keras.all.hit5'] * 100,
        m['keyword_constrained']['keras-pytorch.all.hit5'] * 100,
        m['keyword_constrained']['pytorch-keras.all.mrr'] * 100,
        m['keyword_constrained']['keras-pytorch.all.mrr'] * 100,
        m['translate_pytorch'][1],
        m['translate_keras'][1],
        m['translate_pytorch'][0] * 100,
        m['translate_keras'][0] * 100
    ]


def _run(args):
    if not os.path.exists(args.out_dir) or not os.path.exists(os.path.join(args.out_dir, "log.json")):
        adv_train(args)
    return _evaluate(
        bpe_path='processed_data/codes',
        vocab_path='processed_data/dict.txt',
        adv_root=args.out_dir,
        checkpoint=5,
        eval_data_keyword_path="data/eval_data_keyword.json",
        eval_data_translate_path="data/eval_data_translate.json"
    )


def main():
    df = pd.DataFrame(
        data=np.zeros((2, 10), dtype=np.double),
        columns=["hit1.p", "hit1.k", "hit5.p", "hit5.k", "mrr.p", "mrr.k", "bleu.p", "bleu.k", "f1.p", "f1.k"],
        index=["adelt_small", "adelt_base"]
    )

    lst = []
    for seed in ['10', '20', '30', '40', '50']:
        lst.append(_run(Options(
            dim=2048,
            d_dim_hid=2048,
            drop=0.1,
            leaky=0.2,
            beta1=0.9,
            beta2=0.999,
            wd=0.001,
            epoch=5,
            warmup=0.1,
            d_steps=1,
            d_smooth=0.2,
            data_dir='processed_data/span_states_pybert_small/',
            out_dir=f'adv_checkpoints/pybert_small/seed{seed}',
            dim_in=512,
            do_adv=True,
            lr=0.0002,
            bs=128,
            seed=int(seed)
        )))
    df.loc["adelt_small", :] = np.array(lst).mean(axis=0)

    lst = []
    for seed in ['10', '20', '30', '40', '50']:
        lst.append(_run(Options(
            dim=2048,
            d_dim_hid=2048,
            drop=0.1,
            leaky=0.2,
            beta1=0.9,
            beta2=0.999,
            wd=0.001,
            epoch=5,
            warmup=0.1,
            d_steps=1,
            d_smooth=0.2,
            data_dir='processed_data/span_states_pybert_base/',
            out_dir=f'adv_checkpoints/pybert_base/seed{seed}',
            dim_in=768,
            do_adv=True,
            lr=0.0005,
            bs=256,
            seed=int(seed)
        )))
    df.loc["adelt_base", :] = np.array(lst).mean(axis=0)

    df.to_pickle("result.pkl")


if __name__ == '__main__':
    main()
