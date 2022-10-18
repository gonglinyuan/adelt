# ADELT

Welcome to the repository for ADELT.

Paper: [ADELT: Transpilation Between Deep Learning Frameworks](https://arxiv.org/abs/2303.03593) (IJCAI 2024)

Authors: [Linyuan Gong](https://gonglinyuan.com), Jiayi Wang, Alvin Cheung

ADELT is a novel approach to source-to-source transpilation between deep learning frameworks. ADELT uniquely decouples
code skeleton transpilation and API keyword mapping. For code skeleton transpilation, it uses few-shot prompting on
large language models (LLMs), while for API keyword mapping, it uses contextual embeddings from a code-specific BERT.
These embeddings are trained in a domain-adversarial setup to generate a keyword translation dictionary. ADELT is
trained on an unlabeled web-crawled deep learning corpus, without relying on any hand-crafted rules or parallel data.

## Reproducing ADELT

Requires Python 3.8 and PyTorch 1.7.1. Here is an example script for environment setup:

```bash
conda create -n adelt python=3.8 cudatoolkit=11.0 jupyter jupyterlab
conda activate adelt
python -m pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install tensorflow==2.8.0 asttokunparse editdistance fastbpe gensim hydra-core==1.0.7 pandas sacrebleu==1.2.11 tqdm regex
```

Step 1: preprocess data

```bash
python -m adelt.preprocess_dl \
processed_data/dl_classes.pytorch.tok \
processed_data/dl_classes.keras.tok \
--output_dir processed_data/dl_classes.bin \
--n_train_splits 8 \
--bpe_path processed_data/codes \
--vocab_path processed_data/dict.txt
```

Step 2: extract states from PyBERT

```bash
python -m adelt.data \
--data_dir processed_data/dl_classes.bin \
--lm_path lm_checkpoints/base \
--n_spans 921600 \
--bs 64 \
--bptt 512 \
--out_dir processed_data/span_states_pybert_base
```

```bash
python -m adelt.data \
--data_dir processed_data/dl_classes.bin \
--lm_path lm_checkpoints/small \
--n_spans 921600 \
--bs 64 \
--bptt 512 \
--out_dir processed_data/span_states_pybert_small
```

Step 3: train models to reproduce results

```bash
python run_adv_train.py
```

## Citation

```
@article{
    gong2023adelt,
    title={{ADELT}: Transpilation Between Deep Learning Frameworks},
    url={http://arxiv.org/abs/2303.03593},
    DOI={10.48550/arXiv.2303.03593},
    note={arXiv:2303.03593 [cs]},
    number={arXiv:2303.03593},
    publisher={arXiv},
    author={Gong, Linyuan and Wang, Jiayi and Cheung, Alvin},
    year={2023},
    month=mar
}
```