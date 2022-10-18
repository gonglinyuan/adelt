import argparse
import ast
import os
import shutil

import asttokunparse
import fastBPE
import numpy as np
import torch

import adelt.api_call_extraction as api_call_extraction
import adelt.tokenization as tokenization
from fairseq.data import Dictionary


def span_tokenize(tokens):
    converter = tokenization._TokenConverterPython()
    span_count = {}
    spans = []
    for i, (toktype, tok) in enumerate(tokens):
        is_docstring = asttokunparse.is_docstring(tokens, i)
        if toktype >= 0:
            status = converter.step(toktype, tok, is_docstring)
            if status == converter.EOS:
                break
            elif status == converter.ERROR:
                return [], []
            else:
                assert status == converter.CONTINUE
        elif toktype == asttokunparse.SPAN_OPEN:
            if tok not in span_count:
                span_count[tok] = []
            span_count[tok].append(len(converter.tokens))
        elif toktype == asttokunparse.SPAN_CLOSE:
            last_pos = span_count[tok].pop()
            spans.append((tok, last_pos, len(converter.tokens)))
        else:
            raise ValueError()
    return converter.get_results(), spans


def span_apply_bpe(bpe_model, tokens, spans):
    indices, bpe_tokens = [], []
    for i, joined_tokens in enumerate(bpe_model.apply(tokens)):
        indices.append(len(bpe_tokens))
        bpe_tokens.extend(joined_tokens.split())
    spans = [
        (nm, indices[l], indices[r])
        for nm, l, r in spans
    ]
    return bpe_tokens, spans


def process_example(bpe_model, dico: Dictionary, code, lang, kind_s, span_pos_s, span_lr_s, base_txt=0):
    tree = ast.parse(code)
    extractor = api_call_extraction.ApiCallExtractor(lang)
    extractor.visit(tree)
    py_tokens = asttokunparse.span_unparse(tree, extractor.build_node_to_span())
    tokens, spans = span_tokenize(py_tokens)
    bpe_tokens, spans = span_apply_bpe(bpe_model, tokens, spans)
    indexed = dico.encode_line(" ".join(bpe_tokens), add_if_not_exist=False, append_eos=False).tolist()
    span_to_lr = {
        nm: [base_txt + l, base_txt + r]
        for nm, l, r in spans
    }
    for api_call in extractor.api_calls:
        kind_s.append(api_call.kind)
        span_pos_s.append([len(span_lr_s), len(span_lr_s) + len(api_call.arg_kws) + 2])
        span_lr_s.append(span_to_lr[f"span_{api_call.span}"])
        span_lr_s.append(span_to_lr[f"span_{api_call.api_name}"])
        for arg_kw in api_call.arg_kws:
            span_lr_s.append(span_to_lr[f"span_{arg_kw}"])
    return indexed


def tensorize_data(positions, sentences, dico):
    # tensorize data
    positions = np.int64(positions)
    if len(dico) < 1 << 16:
        sentences = np.uint16(sentences)
    elif len(dico) < 1 << 31:
        sentences = np.int32(sentences)
    else:
        raise Exception("Dictionary is too big.")
    assert sentences.min() >= 0
    return positions, sentences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path_pytorch", type=str)
    parser.add_argument("input_path_keras", type=str)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_train_splits", type=int, required=True)
    parser.add_argument("--bpe_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    shutil.copyfile(args.vocab_path, os.path.join(args.output_dir, "dict.txt"))
    for lang in ['pytorch', 'keras']:
        lines = []
        with open(getattr(args, f"input_path_{lang}"), "r", encoding="utf-8") as fi:
            for line in fi:
                lines.append(line)
        rng = np.random.RandomState(0)
        rng.shuffle(lines)
        bpe_model = fastBPE.fastBPE(args.bpe_path)
        dico = Dictionary.load(args.vocab_path)
        for split_id in range(args.n_train_splits):
            positions = []
            sentences = []
            ac_kind_s, ac_span_pos_s, ac_span_lr_s = [], [], []
            for i in range(split_id, len(lines), args.n_train_splits):
                _, code = lines[i].split("|", 1)
                code = tokenization.detokenize_python(code.strip().split(" "))
                indexed = process_example(
                    bpe_model, dico, code, lang,
                    ac_kind_s, ac_span_pos_s, ac_span_lr_s,
                    len(sentences)
                )
                positions.append([len(sentences), len(sentences) + len(indexed)])
                sentences.extend(indexed)
                sentences.append(dico.eos_index)  # EOS
            positions, sentences = tensorize_data(positions, sentences, dico)
            ac_kind_s = np.int16(ac_kind_s)
            ac_span_pos_s, ac_span_lr_s = np.int64(ac_span_pos_s), np.int64(ac_span_lr_s)
            data = {
                'dico': dico,
                'positions': positions,
                'sentences': sentences,
                'ac_kind_s': ac_kind_s,
                'ac_span_pos_s': ac_span_pos_s,
                'ac_span_lr_s': ac_span_lr_s,
            }
            torch.save(data, os.path.join(args.output_dir, f"train.{lang}.{split_id}.pth"))


if __name__ == '__main__':
    main()
