import ast
import itertools

import astunparse

from adelt.api_call_extraction import ApiCallInfo
from adelt.canonicalization import ModuleCanonicalizer, AliasCanonicalizer, ApiCallCanonicalizer
from adelt.preprocess_dl import process_example
from adelt.tokenization import detokenize_python
from .evaluator import Evaluator
from .utils import OTHER_LANG


class TranslationCleanup(ast.NodeTransformer):
    def visit_Call(self, node):
        node.keywords = [
            keyword
            for keyword in node.keywords
            if not (keyword.arg and keyword.arg.startswith("KEYWORD_ARGUMENT_TO_DROP"))
        ]
        return node


def canonicalize(code: str) -> str:
    tree = ast.parse(code)
    tree = ModuleCanonicalizer().visit(tree)
    tree = AliasCanonicalizer().visit(tree)
    tree = ApiCallCanonicalizer().visit(tree)
    return astunparse.unparse(tree).strip()


def translate(evaluator: Evaluator, keyword_match, code: str, lang: str) -> str:
    code = canonicalize(code)
    kind_s, span_pos_s, span_lr_s = [], [], []
    indexed = process_example(evaluator.bpe_model, evaluator.dico, code, lang, kind_s, span_pos_s, span_lr_s)
    indexed = indexed + [evaluator.dico.eos_index]

    replace_table = []
    for i, (kind, (pos_l, pos_r)) in enumerate(zip(kind_s, span_pos_s)):
        l, r = span_lr_s[pos_l + 1]
        api_name = indexed[l: r]
        kw_desc = kind, tuple(api_name)
        kw_id = evaluator.kwdesc_to_id[lang].get(kw_desc, -1)
        if kw_id == -1:
            continue
        replace_table.append((l, r, kw_id, i))
        for l, r in span_lr_s[pos_l + 2: pos_r]:
            kw_desc = ApiCallInfo.KIND_KEYWORD, tuple(api_name), tuple(indexed[l: r])
            kw_id = evaluator.kwdesc_to_id[lang].get(kw_desc, -1)
            if kw_id == -1:
                continue
            replace_table.append((l, r, kw_id, i))

    replace_table = sorted(replace_table)
    for i in range(1, len(replace_table)):
        assert replace_table[i - 1][1] <= replace_table[i][0]

    keyword_match_dic = dict(keyword_match[lang])
    replace_table = [(l, r, keyword_match_dic.get(x, -1), i) for l, r, x, i in replace_table]

    target_tokens = []
    last = len(indexed)
    used_kw = set()
    drop_cnt = 0
    for l, r, y, i in reversed(replace_table):
        target_tokens.append([evaluator.dico[wi] for wi in indexed[r: last]])
        if y == -1 or (i, y) in used_kw:
            target_tokens.append([f"KEYWORD_ARGUMENT_TO_DROP_{drop_cnt}"])
            drop_cnt += 1
        else:
            target_tokens.append([evaluator.dico[wi] for wi in evaluator.id_to_kwdesc[OTHER_LANG[lang]][y][-1]])
            used_kw.add((i, y))
        last = l
    target_tokens.append([evaluator.dico[wi] for wi in indexed[0: last]])
    target_tokens = list(itertools.chain(*reversed(target_tokens)))
    while target_tokens and target_tokens[-1] == evaluator.dico[evaluator.dico.eos_index]:
        target_tokens.pop()

    code = " ".join(target_tokens).replace("@@ ", "")
    code = code.strip().split(" ")
    code = detokenize_python(code)
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(code)
        print(replace_table)
        raise e
    tree = TranslationCleanup().visit(tree)
    return canonicalize(astunparse.unparse(tree)).strip()


def translate_all(evaluator, keyword_match):
    p_hypos, p_refs, p_srcs = [], [], []
    k_hypos, k_refs, k_srcs = [], [], []
    for p_s, k_s, do_p, do_k in evaluator.translate_data:
        if do_p:
            p_hypos.append(translate(evaluator, keyword_match, p_s, "pytorch"))
            p_refs.append(canonicalize(k_s))
            p_srcs.append(canonicalize(p_s))
        if do_k:
            k_hypos.append(translate(evaluator, keyword_match, k_s, "keras"))
            k_refs.append(canonicalize(p_s))
            k_srcs.append(canonicalize(k_s))
    return (p_srcs, p_refs, p_hypos), (k_srcs, k_refs, k_hypos)
