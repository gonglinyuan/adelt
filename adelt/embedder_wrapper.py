from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import torch
from gensim.models import KeyedVectors

from adelt.api_call_extraction import ApiCallInfo
from fairseq.models.roberta import RobertaHubInterface, RobertaModel


def acs_to_spans(acs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], *, api_call, api_name, api_kws):
    spans = []
    for kind_s, span_pos_s, span_lr_s in acs:
        cur_spans = []
        for i in range(kind_s.shape[0]):
            st, ed = span_pos_s[i]
            for j in range(ed - st):
                if j == 0 and api_call:
                    cur_spans.append((span_lr_s[st + j, 0], span_lr_s[st + j, 1], -1))
                elif j == 1 and api_name:
                    cur_spans.append((span_lr_s[st + j, 0], span_lr_s[st + j, 1], kind_s[i]))
                elif j > 1 and api_kws:
                    cur_spans.append((span_lr_s[st + j, 0], span_lr_s[st + j, 1], ApiCallInfo.KIND_KEYWORD))
        spans.append(cur_spans)
    return spans


@dataclass(frozen=True)
class ApiCallEmbedding:
    kind: int
    span: Optional[Tuple[torch.FloatTensor, torch.LongTensor]]
    api_name: Optional[Tuple[torch.FloatTensor, torch.LongTensor]]
    arg_kws: List[Tuple[torch.FloatTensor, torch.LongTensor]]


def regroup_acs(acs: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], span_states, *, api_call, api_name, api_kws):
    ac_embeds = []
    assert len(acs) == len(span_states)
    for (kind_s, span_pos_s, span_lr_s), cur_span_states in zip(acs, span_states):
        span_id = 0
        cur_ac_embeds = []
        for i in range(kind_s.shape[0]):
            ac_kind = None
            span_embed = None
            api_name_embed = None
            arg_kws_embeds = []
            st, ed = span_pos_s[i]
            for j in range(ed - st):
                if j == 0 and api_call:
                    state, y_src, kind = cur_span_states[span_id]
                    span_id += 1
                    assert kind == -1
                    span_embed = (state, y_src)
                elif j == 1 and api_name:
                    state, y_src, kind = cur_span_states[span_id]
                    span_id += 1
                    assert kind == kind_s[i]
                    api_name_embed = (state, y_src)
                    ac_kind = kind
                elif j > 1 and api_kws:
                    state, y_src, kind = cur_span_states[span_id]
                    span_id += 1
                    assert kind == ApiCallInfo.KIND_KEYWORD
                    arg_kws_embeds.append((state, y_src))
            cur_ac_embeds.append(ApiCallEmbedding(ac_kind, span_embed, api_name_embed, arg_kws_embeds))
        assert span_id == len(cur_span_states)
        ac_embeds.append(cur_ac_embeds)
    return ac_embeds


class EmbedderWrapper:
    @staticmethod
    def maybe_regroup(ac1, span_states, regroup):
        if regroup:
            return regroup_acs(ac1, span_states, api_call=False, api_name=True, api_kws=True)
        else:
            return span_states

    def embed(self, x1, len1, ac1, lang, regroup=False):
        raise NotImplementedError()


class SpanEmbedder:
    def __init__(self, model: RobertaHubInterface):
        self.model = model
        self.model.cuda()
        self.model.eval()

    def get_embeddings(self, tokens, spans):
        slen, bs = tokens.size()
        features = self.model.extract_features(tokens.transpose(0, 1))
        features = features.transpose(0, 1)
        assert features.size(0) == slen and features.size(1) == bs
        return [
            [
                (features[l: r, bi, :].cpu(), tokens[l: r, bi].cpu(), t)
                for l, r, t in spans[bi]
            ]
            for bi in range(len(spans))
        ]


class TransformerEmbedderWrapper(EmbedderWrapper):
    def __init__(self, lm_path):
        self.span_embedder = SpanEmbedder(RobertaModel.from_pretrained(lm_path))

    def embed(self, x1, len1, ac1, lang, regroup=False):
        x1 = x1.cuda()
        spans1 = acs_to_spans(ac1, api_call=False, api_name=True, api_kws=True)
        with torch.no_grad():
            span_states = self.span_embedder.get_embeddings(x1, spans1)
            return self.maybe_regroup(ac1, span_states, regroup)


class Word2VecEmbedderWrapper(EmbedderWrapper):
    def __init__(self, lm_path, dico):
        model = KeyedVectors.load_word2vec_format(lm_path, binary=True)
        self.emb = torch.zeros((len(dico), model.vector_size), dtype=torch.float)
        for key in model.index_to_key:
            vec = torch.from_numpy(model.get_vector(key))
            if key == "</s>":
                self.emb[dico.eos_index, :] = vec
            else:
                self.emb[int(key), :] = vec

    def embed(self, x1, len1, ac1, lang, regroup=False):
        spans = acs_to_spans(ac1, api_call=False, api_name=True, api_kws=True)
        span_states = [
            [
                (self.emb[x1[l:r, bi], :], x1[l: r, bi], t)
                for l, r, t in spans[bi]
            ]
            for bi in range(len(spans))
        ]
        return self.maybe_regroup(ac1, span_states, regroup)
