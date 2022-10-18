import ast
import itertools
import json
import os
import subprocess
import tempfile

import asttokunparse
import editdistance
import fastBPE
import torch
from IPython.display import display, Markdown

import adelt.tokenization as tokenization
from adelt.api_call_extraction import ApiCallInfo
from adelt.call_extraction import CallExtractor
from adelt.preprocess_dl import span_tokenize, span_apply_bpe
from adelt.utils import OTHER_LANG
from fairseq.data.dictionary import Dictionary

METRICS = ["mr", "mrr", "hit1", "hit5"]
BLEU_SCRIPT_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../tools/multi-bleu.perl')


class Evaluator:
    def __init__(self, bpe_model, dico, adv_ckpt, api_data=None, translate_data=None, edit_distance=None):
        self.bpe_model = bpe_model
        self.dico = dico
        self.adv_ckpt = adv_ckpt
        self.id_to_kwdesc = adv_ckpt['id_to_kwdesc']
        self.kwdesc_to_id = {}
        for lang in ['pytorch', 'keras']:
            self.kwdesc_to_id[lang] = {w: i for i, w in enumerate(self.id_to_kwdesc[lang])}
        if api_data is not None:
            self.api_data = api_data
            self.encoded_api_data = {'func': (self._encode_eval_data())[0], 'arg': (self._encode_eval_data())[1]}
        self.score_mat = (
            self._generate_edit_distance_matrix(edit_distance)
            if edit_distance is not None
            else self._generate_keyword_mapping_matrix()
        )
        if translate_data is not None:
            self.translate_data = translate_data

    @classmethod
    def load(cls, bpe_path, vocab_path, adv_root, checkpoint,
             api_data_path=None, translate_data_path=None, edit_distance=None):
        bpe_model = fastBPE.fastBPE(bpe_path)
        dico = Dictionary.load(vocab_path)
        adv_ckpt = torch.load(os.path.join(adv_root, f'checkpoint_{checkpoint}.pt'))
        if api_data_path is not None:
            with open(api_data_path, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
        else:
            eval_data = None
        if translate_data_path is not None:
            with open(translate_data_path, "r", encoding="utf-8") as f:
                translate_data = json.load(f)
        else:
            translate_data = None
        return cls(bpe_model, dico, adv_ckpt, eval_data, translate_data, edit_distance)

    def _encode_str(self, s):
        s = tokenization.tokenize_python(s)[:-1]
        # s = itertools.chain(*[t.split() for t in self.bpe_model.apply(s)])
        # s = tuple(self.dico.index(w) for w in s)
        s = " ".join(self.bpe_model.apply(s))
        s = tuple(self.dico.encode_line(s, add_if_not_exist=False, append_eos=False).tolist())
        assert "<unk>" not in s
        return s

    def _decode_ids(self, s):
        s = [self.dico[wi] for wi in s]
        s = " ".join(s).replace("@@ ", "")
        s = s.strip().split(" ")
        s = tokenization.detokenize_python(s)
        return s

    def _encode_eval_data(self):
        def _process_single(x, lang):
            if x is None:
                return -1, None
            if x[0] == 'LAYER':
                class_desc = ApiCallInfo.KIND_LAYER, self._encode_str(x[1])
            elif x[0] == 'FUNC':
                class_desc = ApiCallInfo.KIND_FUNC, self._encode_str(x[1])
            elif x[0] == 'KEYWORD':
                class_desc = ApiCallInfo.KIND_KEYWORD, self._encode_str(x[1]), self._encode_str(x[2])
            else:
                raise NotImplementedError
            return self.kwdesc_to_id[lang][class_desc], class_desc[0]

        result_func, result_arg = {'pytorch': [], 'keras': []}, {'pytorch': [], 'keras': []}
        for u, v in self.api_data:
            u, ty_u = _process_single(u, 'pytorch')
            v, ty_v = _process_single(v, 'keras')
            ty = ty_u if ty_u is not None else ty_v
            assert ty is not None and (ty_v is None or ty == ty_v)
            result = (result_arg if ty == ApiCallInfo.KIND_KEYWORD else result_func)
            if u >= 0:
                result['pytorch'].append((u, v))
            if v >= 0:
                result['keras'].append((v, u))
        return result_func, result_arg

    def _generate_edit_distance_matrix(self, s_type):
        score_mat = {}
        score_mat['edit'] = torch.zeros(
            (len(self.id_to_kwdesc['pytorch']) + 1, len(self.id_to_kwdesc['keras']) + 1),
            dtype=torch.double
        )
        for p_id, p_kwdesc in enumerate(self.id_to_kwdesc['pytorch']):
            for k_id, k_kwdesc in enumerate(self.id_to_kwdesc['keras']):
                p_str = self._decode_ids(p_kwdesc[-1]).strip()
                k_str = self._decode_ids(k_kwdesc[-1]).strip()
                if s_type == "uncased":
                    p_str = p_str.lower()
                    k_str = k_str.lower()
                score_mat['edit'][p_id, k_id] = float(-editdistance.eval(p_str, k_str))
        score_mat['edit'][:-1, -1] = score_mat['edit'][:-1, :-1].mean(dim=1)
        score_mat['edit'][-1, :-1] = score_mat['edit'][:-1, :-1].mean(dim=0)

        p_avg_top_5 = score_mat['edit'].topk(5, dim=0).values.mean(dim=0, keepdims=True)
        k_avg_top_5 = score_mat['edit'].topk(5, dim=1).values.mean(dim=1, keepdims=True)
        score_mat['csls_5'] = (
            score_mat['edit'] * 2
            - p_avg_top_5.expand_as(score_mat['edit'])
            - k_avg_top_5.expand_as(score_mat['edit'])
        )
        p_avg_top_10 = score_mat['edit'].topk(10, dim=0).values.mean(dim=0, keepdims=True)
        k_avg_top_10 = score_mat['edit'].topk(10, dim=1).values.mean(dim=1, keepdims=True)
        score_mat['csls_10'] = (
            score_mat['edit'] * 2
            - p_avg_top_10.expand_as(score_mat['edit'])
            - k_avg_top_10.expand_as(score_mat['edit'])
        )
        return score_mat

    def _generate_keyword_mapping_matrix(self):
        dim = self.adv_ckpt['args']['dim']
        p_mat = self.adv_ckpt['generator']['classifier_pytorch.1.weight']
        k_mat = self.adv_ckpt['generator']['classifier_keras.1.weight']
        p_mat = torch.cat([p_mat, p_mat.new_zeros((1, dim))], dim=0).double()
        k_mat = torch.cat([k_mat, k_mat.new_zeros((1, dim))], dim=0).double()
        score_mat = {}

        score_mat['inner'] = torch.mm(p_mat, k_mat.T)
        p_mat_norm = p_mat / (p_mat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        k_mat_norm = k_mat / (k_mat.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        score_mat['cos'] = torch.mm(p_mat_norm, k_mat_norm.T)
        p_avg_top_5 = score_mat['cos'].topk(5, dim=0).values.mean(dim=0, keepdims=True)
        k_avg_top_5 = score_mat['cos'].topk(5, dim=1).values.mean(dim=1, keepdims=True)
        score_mat['csls_5'] = (
            score_mat['cos'] * 2
            - p_avg_top_5.expand_as(score_mat['cos'])
            - k_avg_top_5.expand_as(score_mat['cos'])
        )
        p_avg_top_10 = score_mat['cos'].topk(10, dim=0).values.mean(dim=0, keepdims=True)
        k_avg_top_10 = score_mat['cos'].topk(10, dim=1).values.mean(dim=1, keepdims=True)
        score_mat['csls_10'] = (
            score_mat['cos'] * 2
            - p_avg_top_10.expand_as(score_mat['cos'])
            - k_avg_top_10.expand_as(score_mat['cos'])
        )
        return score_mat

    @staticmethod
    def _compute_metrics(scores, idx):
        rank = ((scores > scores[idx]).sum() + 1).item()
        return {
            "mr": rank,
            "mrr": 1.0 / rank,
            "hit1": int(rank == 1),
            "hit5": int(rank <= 5)
        }

    def evaluate_keyword_unsupervised(self, keyword_match, top_k):
        results = {"pytorch-keras.mean_cos": 0.0, "keras-pytorch.mean_cos": 0.0}

        keyword_match_dic = {
            "pytorch": dict(keyword_match["pytorch"]),
            "keras": dict(keyword_match["keras"])
        }
        for u in range(top_k):
            v_pred = keyword_match_dic["pytorch"].get(u, -1)
            u_pred = keyword_match_dic["keras"].get(v_pred, -1)
            if u_pred == u:
                results["pytorch-keras.mean_cos"] += self.score_mat["cos"][u, v_pred].item()
        results["pytorch-keras.mean_cos"] /= top_k
        for u in range(top_k):
            v_pred = keyword_match_dic["keras"].get(u, -1)
            u_pred = keyword_match_dic["pytorch"].get(v_pred, -1)
            if u_pred == u:
                results["keras-pytorch.mean_cos"] += self.score_mat["cos"][v_pred, u].item()
        results["keras-pytorch.mean_cos"] /= top_k

        return results

    def evaluate_keyword_unconstrained(self):
        assert self.encoded_api_data is not None

        results = {}
        m_sum, m_cnt = {}, {}

        for ty in ['func', 'arg', 'all']:
            for s_type in self.score_mat.keys():
                for metric in METRICS:
                    m_sum[f"{s_type}.{ty}.{metric}"] = 0.0
                    m_cnt[f"{s_type}.{ty}.{metric}"] = 0
        for ty in ['func', 'arg']:
            for p_id, k_id in self.encoded_api_data[ty]['pytorch']:
                for s_type in self.score_mat.keys():
                    scores = self.score_mat[s_type][p_id, :]
                    for metric, v in self._compute_metrics(scores, k_id).items():
                        m_sum[f"{s_type}.{ty}.{metric}"] += v
                        m_cnt[f"{s_type}.{ty}.{metric}"] += 1
                        m_sum[f"{s_type}.all.{metric}"] += v
                        m_cnt[f"{s_type}.all.{metric}"] += 1
        for ty in ['func', 'arg', 'all']:
            for s_type in self.score_mat.keys():
                for metric in METRICS:
                    results[f"{s_type}.pytorch-keras.{ty}.{metric}"] = (
                        m_sum[f"{s_type}.{ty}.{metric}"] / m_cnt[f"{s_type}.{ty}.{metric}"])

        for ty in ['func', 'arg', 'all']:
            for s_type in self.score_mat.keys():
                for metric in METRICS:
                    m_sum[f"{s_type}.{ty}.{metric}"] = 0.0
                    m_cnt[f"{s_type}.{ty}.{metric}"] = 0
        for ty in ['func', 'arg']:
            for k_id, p_id in self.encoded_api_data[ty]['keras']:
                for s_type in self.score_mat.keys():
                    scores = self.score_mat[s_type][:, k_id]
                    for metric, v in self._compute_metrics(scores, p_id).items():
                        m_sum[f"{s_type}.{ty}.{metric}"] += v
                        m_cnt[f"{s_type}.{ty}.{metric}"] += 1
                        m_sum[f"{s_type}.all.{metric}"] += v
                        m_cnt[f"{s_type}.all.{metric}"] += 1
        for ty in ['func', 'arg', 'all']:
            for s_type in self.score_mat.keys():
                for metric in METRICS:
                    results[f"{s_type}.keras-pytorch.{ty}.{metric}"] = (
                        m_sum[f"{s_type}.{ty}.{metric}"] / m_cnt[f"{s_type}.{ty}.{metric}"])

        return results

    def _display_kwdesc(self, lang, kw_id):
        if kw_id == -1:
            return "`None`"
        kw_desc = self.id_to_kwdesc[lang][kw_id]
        if kw_desc[0] == ApiCallInfo.KIND_LAYER or kw_desc[0] == ApiCallInfo.KIND_FUNC:
            return f"`{self._decode_ids(kw_desc[1])}`"
        elif kw_desc[0] == ApiCallInfo.KIND_KEYWORD:
            return f"`{self._decode_ids(kw_desc[2])}` (`{self._decode_ids(kw_desc[1])}`)"
        else:
            raise NotImplementedError()

    def evaluate_keyword_constrained(self, keyword_match, p_rank, k_rank, quiet=False):
        assert self.encoded_api_data is not None
        results = {}
        for lang in ['pytorch', 'keras']:
            if not quiet:
                display(Markdown(f"## {lang} -> {OTHER_LANG[lang]}"))
            m_sum, m_cnt = {}, {}
            match_dic = dict(keyword_match[lang])
            rank_mat = (p_rank if lang == 'pytorch' else k_rank.t())
            for ty in ['func', 'arg', 'all']:
                for metric in METRICS:
                    m_sum[f"{ty}.{metric}"] = 0.0
                    m_cnt[f"{ty}.{metric}"] = 0
            for ty in ['func', 'arg']:
                for u, v in self.encoded_api_data[ty][lang]:
                    v_pred = match_dic.get(u, -1)
                    assert rank_mat[u, v_pred].item() == 0
                    for metric, w in self._compute_metrics(-rank_mat[u, :], v).items():
                        m_sum[f"{ty}.{metric}"] += w
                        m_cnt[f"{ty}.{metric}"] += 1
                        m_sum[f"all.{metric}"] += w
                        m_cnt[f"all.{metric}"] += 1
                    if not quiet:
                        display(Markdown(
                            f"Source: {self._display_kwdesc(lang, u)}    "
                            f"Truth: {self._display_kwdesc(OTHER_LANG[lang], v)}    "
                            f"Hypo: {self._display_kwdesc(OTHER_LANG[lang], v_pred)}    "
                            + ("✔️" if v_pred == v else "❌")
                        ))
            for ty in ['func', 'arg', 'all']:
                for metric in METRICS:
                    results[f"{lang}-{OTHER_LANG[lang]}.{ty}.{metric}"] = (
                        m_sum[f"{ty}.{metric}"] / m_cnt[f"{ty}.{metric}"])
        return results

    def _get_call_tokens(self, code, lang):
        tree = ast.parse(code)
        extractor = CallExtractor(lang)
        extractor.visit(tree)
        py_tokens = asttokunparse.span_unparse(tree, extractor.build_node_to_span())
        tokens, spans = span_tokenize(py_tokens)
        bpe_tokens, spans = span_apply_bpe(self.bpe_model, tokens, spans)
        indexed = self.dico.encode_line(" ".join(bpe_tokens), add_if_not_exist=False, append_eos=True).tolist()
        span_to_lr = {nm: [l, r] for nm, l, r in spans}
        result = []
        for call in extractor.calls:
            l, r = span_to_lr[f"span_{call}"]
            result.append(tuple(indexed[l: r]))
        return result

    def evaluate_translate(self, srcs, refs, hypos, lang, quiet=False):
        assert len(refs) == len(hypos)

        # compute F1
        other_lang = OTHER_LANG[lang]
        total_f1 = 0
        for src, ref, hypo in zip(srcs, refs, hypos):
            ref_calls = set(self._get_call_tokens(ref, other_lang))
            hypo_calls = set(self._get_call_tokens(hypo, other_lang))
            common = ref_calls & hypo_calls
            if len(ref_calls) + len(hypo_calls) > 0:
                f1 = 2.0 * len(common) / (len(ref_calls) + len(hypo_calls))
            else:
                f1 = 1.0
            if not quiet:
                display(Markdown(
                    f"Source:\n```python\n{src}\n```\n\n"
                    f"Truth:\n```python\n{ref}\n```\n\n"
                    f"Hypo:\n```python\n{hypo}\n```\n\n"
                    f"Matched: {len(common)}. Ref #: {len(ref_calls)}. Hypo #: {len(hypo_calls)}. F1: {f1}."
                ))
            total_f1 += f1
        total_f1 = total_f1 / len(refs)

        # compute BLEU
        refs_tok = [tokenization.tokenize_python(ref) for ref in refs]
        hypos_tok = [tokenization.tokenize_python(hypo) for hypo in hypos]
        with tempfile.NamedTemporaryFile("w", encoding="utf-8") as refs_fp:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8") as hypos_fp:
                for ref in refs_tok:
                    refs_fp.write(" ".join(ref) + "\n")
                refs_fp.flush()
                for hypo in hypos_tok:
                    hypos_fp.write(" ".join(hypo) + "\n")
                hypos_fp.flush()
                assert os.path.isfile(refs_fp.name) and os.path.isfile(hypos_fp.name)
                assert os.path.isfile(BLEU_SCRIPT_PATH)
                process = subprocess.Popen(
                    f"perl {BLEU_SCRIPT_PATH} {refs_fp.name} < {hypos_fp.name}",
                    stdout=subprocess.PIPE, shell=True
                )
                stdout, _ = process.communicate()
                stdout.decode("utf-8")
                result = stdout.decode("utf-8")
                if result.startswith('BLEU'):
                    bleu = float(result[7:result.index(',')])
                else:
                    print("Impossible to parse BLEU score")
                    bleu = None
        return total_f1, bleu
