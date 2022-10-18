import argparse
import math
import os
from collections import Counter
from glob import glob

import numpy as np
import torch
from tqdm import tqdm

from adelt.api_call_extraction import ApiCallInfo
from adelt.embedder_wrapper import TransformerEmbedderWrapper, Word2VecEmbedderWrapper
from fairseq.data.dictionary import Dictionary


class StreamSpanDataset:
    def __init__(self, sent, pos, ac_kind_s, ac_span_pos_s, ac_span_lr_s, bs, bptt, eos, bos):
        self.bs = bs
        self.bptt = bptt
        self.eos = eos
        self.bos = bos

        # checks
        assert len(pos) == (sent == self.eos).sum()
        assert len(pos) == (sent[pos[:, 1]] == self.eos).sum()

        self.sent = sent
        self.n_tokens = len(sent)
        self.dtype = self.sent.dtype
        self.lengths = torch.LongTensor(bs).fill_(self.bptt)

        self.ac_kind_s = ac_kind_s
        self.ac_span_pos_s = ac_span_pos_s
        self.ac_span_lr_s = ac_span_lr_s
        assert len(ac_kind_s) == len(ac_span_pos_s)
        assert ac_span_pos_s[-1, 1] == len(ac_span_lr_s)

    def get_iterator(self, shuffle, seed=None):
        rng = np.random.RandomState(seed)
        offset = rng.randint(0, self.bptt - 1)
        n_seqs = int(math.ceil((self.n_tokens + offset) / (self.bptt - 1)))
        t_size = n_seqs * (self.bptt - 1)
        buffer = np.zeros(t_size, dtype=self.dtype) + self.eos
        buffer[offset: offset + self.n_tokens] = self.sent
        buffer = buffer.reshape(n_seqs, self.bptt - 1)

        ij_lr = self.ac_span_lr_s + offset
        i_lr = ij_lr // (self.bptt - 1)
        j_lr = ij_lr % (self.bptt - 1) + 1
        matched: np.ndarray = i_lr[:, 0] == i_lr[:, 1]
        api_calls = [([], [], []) for _ in range(n_seqs)]
        tmp_count = [0 for _ in range(n_seqs)]
        for idx in range(len(self.ac_kind_s)):
            if matched[self.ac_span_pos_s[idx, 0]]:  # if the whole api call is in a seq
                seq_id = i_lr[self.ac_span_pos_s[idx, 0], 0]
                api_calls[seq_id][0].append(self.ac_kind_s[idx])
                api_calls[seq_id][1].append([
                    tmp_count[seq_id],
                    tmp_count[seq_id] + self.ac_span_pos_s[idx, 1] - self.ac_span_pos_s[idx, 0]
                ])
                api_calls[seq_id][2].append(j_lr[self.ac_span_pos_s[idx, 0]: self.ac_span_pos_s[idx, 1]])
                tmp_count[seq_id] += self.ac_span_pos_s[idx, 1] - self.ac_span_pos_s[idx, 0]
        del tmp_count
        api_calls = [
            (np.int16(ac_kind_s), np.int64(ac_span_pos_s), np.concatenate(ac_span_lr_s, axis=0)) if ac_kind_s else None
            for ac_kind_s, ac_span_pos_s, ac_span_lr_s in api_calls
        ]

        indexes = (rng.permutation if shuffle else range)(n_seqs)
        batch = []
        for seq_id in indexes:
            if api_calls[seq_id] is not None:
                batch.append(seq_id)
                if len(batch) == self.bs:
                    data_x = np.zeros((self.bs, self.bptt), dtype=self.dtype) + self.eos
                    data_api_calls = []
                    for j, bi in enumerate(batch):
                        data_x[j, 1:] = buffer[bi, :]
                        data_api_calls.append(api_calls[bi])
                    data_x[:, 0] = self.bos
                    yield torch.from_numpy(data_x.T.astype(np.int64)), self.lengths, data_api_calls
                    batch = []


class DataManager:
    def __init__(self, data, id_to_kwdesc, bs: int = 1):
        assert len(data['pytorch']) == len(data['keras'])
        assert len(data['pytorch']) % bs == 0
        self.data = data
        self.id_to_kwdesc = id_to_kwdesc

    @classmethod
    def load(cls, data_dir, bs):
        data = torch.load(os.path.join(data_dir, 'data.pt'), map_location='cpu')
        id_to_kwdesc = torch.load(os.path.join(data_dir, 'id_to_kwdesc.pt'), map_location='cpu')
        return cls(data, id_to_kwdesc, bs)

    def save(self, data_dir):
        os.makedirs(data_dir, exist_ok=True)
        torch.save(self.data, os.path.join(data_dir, 'data.pt'), pickle_protocol=4)
        torch.save(self.id_to_kwdesc, os.path.join(data_dir, 'id_to_kwdesc.pt'))

    @classmethod
    def build_for_adv(cls, data_dir, lm_path, n_spans, bs, bptt, lm_type="transformer"):

        def _add_span_state(span_states, counter, pbar, state, kw_desc):
            span_states.append((state, kw_desc))
            counter[kw_desc] += 1
            pbar.update(1)

        print("Creating dataset for adversarial training")
        embedder_wrapper = None
        span_states_s = {}
        counters = {}
        dico = Dictionary.load(os.path.join(data_dir, "dict.txt"))
        for lang in ['pytorch', 'keras']:
            counters[lang] = Counter()
            span_states_s[lang] = []
            epoch = 0
            pbar = tqdm(total=n_spans)
            while len(span_states_s[lang]) < n_spans:
                print('epoch', epoch)
                for mono_data_path in sorted(glob(os.path.join(data_dir, f'train.{lang}.*.pth'))):
                    print(f'Working on {mono_data_path}')
                    mono_data = torch.load(mono_data_path)
                    if embedder_wrapper is None:
                        if lm_type == "transformer":
                            embedder_wrapper = TransformerEmbedderWrapper(lm_path)
                        elif lm_type == "word2vec":
                            embedder_wrapper = Word2VecEmbedderWrapper(lm_path, dico)
                        else:
                            raise ValueError(lm_type)
                    dataset = StreamSpanDataset(
                        mono_data['sentences'], mono_data['positions'],
                        mono_data['ac_kind_s'], mono_data['ac_span_pos_s'], mono_data['ac_span_lr_s'],
                        bs, bptt, dico.eos_index, dico.bos_index
                    )
                    for x1, len1, ac1 in dataset.get_iterator(shuffle=True, seed=epoch):
                        ac_embed_s = embedder_wrapper.embed(x1, len1, ac1, lang, regroup=True)
                        for seq_embeds in ac_embed_s:
                            for ac_embed in seq_embeds:
                                api_name_state, api_name_ids = ac_embed.api_name
                                api_name_ids = tuple(api_name_ids.long().tolist())
                                _add_span_state(
                                    span_states_s[lang], counters[lang], pbar,
                                    api_name_state, (int(ac_embed.kind), api_name_ids)
                                )
                                for arg_kw_state, arg_kw_ids in ac_embed.arg_kws:
                                    arg_kw_ids = tuple(arg_kw_ids.long().tolist())
                                    _add_span_state(
                                        span_states_s[lang], counters[lang], pbar,
                                        arg_kw_state, (ApiCallInfo.KIND_KEYWORD, api_name_ids, arg_kw_ids)
                                    )
                        if len(span_states_s[lang]) >= n_spans:
                            break
                    if len(span_states_s[lang]) >= n_spans:
                        break
                epoch += 1
            pbar.close()
            span_states_s[lang] = span_states_s[lang][:n_spans]

        print("Building keyword indices")
        id_to_kwdesc, kwdesc_to_id = {}, {}
        for lang in ['pytorch', 'keras']:
            id_to_kwdesc[lang], kwdesc_to_id[lang] = [], {}
            for w, _ in sorted(counters[lang].most_common(), key=lambda x: (-x[1], x[0])):
                kwdesc_to_id[lang][w] = len(id_to_kwdesc[lang])
                id_to_kwdesc[lang].append(w)
            print(f"Collected {len(id_to_kwdesc[lang])} distinct keywords for {lang}")

        print("Indexing states")
        for lang in ['pytorch', 'keras']:
            span_states_s[lang] = [
                (state, kwdesc_to_id[lang][class_desc])
                for state, class_desc in span_states_s[lang]
            ]

        return cls(span_states_s, id_to_kwdesc)

    def collate_cuda(self, items1, items2):
        assert len(items1) == len(items2)
        assert items1[0][0].size(1) == items2[0][0].size(1)
        bs = len(items1)
        dim = items1[0][0].size(1)
        dtype = items1[0][0].dtype
        max_len = max(x.size(0) for x, _ in items1 + items2)
        batch_x = torch.zeros((bs * 2, max_len, dim), dtype=dtype, device="cuda")
        batch_mask = torch.zeros((bs * 2, max_len), dtype=dtype, device="cuda")
        batch_y1, batch_y2 = [], []
        for i, (x, y) in enumerate(items1):
            batch_x[i, :x.size(0), :] = x
            batch_mask[i, :x.size(0)] = 1.0
            batch_y1.append(y)
        for i, (x, y) in enumerate(items2):
            batch_x[bs + i, :x.size(0), :] = x
            batch_mask[bs + i, :x.size(0)] = 1.0
            batch_y2.append(y)
        batch_y1 = torch.tensor(batch_y1, dtype=torch.long, device=batch_x.device)
        batch_y2 = torch.tensor(batch_y2, dtype=torch.long, device=batch_x.device)
        return batch_x, batch_mask, batch_y1, batch_y2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--lm_path", type=str, required=True)
    parser.add_argument("--lm_type", type=str, choices=["transformer", "word2vec"], default="transformer")
    parser.add_argument("--n_spans", type=int, required=True)
    parser.add_argument("--bs", type=int, required=True)
    parser.add_argument("--bptt", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    dm = DataManager.build_for_adv(args.data_dir, args.lm_path, args.n_spans, args.bs, args.bptt, args.lm_type)
    dm.save(args.out_dir)


if __name__ == '__main__':
    main()
