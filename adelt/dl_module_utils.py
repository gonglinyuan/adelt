import inspect
from typing import Optional

import tensorflow as tf
import tensorflow.keras.layers as layers
import torch
import torch.nn as nn
import torch.nn.functional as F

NO_SIGNATURES = set()
VAR_SIGNATURES = set()
RESOLVED_ALIAS = set()


def get_signature(name: str, attr: str) -> Optional[inspect.Signature]:
    if name == 'nn':
        if attr in ["RNN", "LSTM", "GRU"]:
            sig = nn.RNNBase
            sig = inspect.signature(sig)
            sig = inspect.Signature([
                v
                for k, v in sig.parameters.items()
                if k != 'mode'
            ])
        else:
            if not hasattr(nn, attr):
                return None
            sig = getattr(nn, attr)
            try:
                sig = inspect.signature(sig)
            except (ValueError, TypeError):
                sig = None
    elif name == 'F':
        if not hasattr(F, attr):
            return None
        sig = getattr(F, attr)
        try:
            sig = inspect.signature(sig)
        except (ValueError, TypeError):
            sig = None
    elif name == 'layers':
        if not hasattr(layers, attr):
            return None
        sig = getattr(layers, attr).__init__
        try:
            sig = inspect.signature(sig)
        except (ValueError, TypeError):
            sig = None
        sig = inspect.Signature([
            v
            for k, v in sig.parameters.items()
            if k != 'self'
        ])
    elif name == 'torch':
        if not hasattr(torch, attr):
            return None
        sig = getattr(torch, attr)
        try:
            sig = inspect.signature(sig)
        except (ValueError, TypeError):
            sig = None
        if sig is None:
            obj = getattr(torch, attr)
            try:
                sig_str = inspect.getdoc(obj).splitlines()[0]
                if sig_str.startswith(attr + "("):
                    if "->" in sig_str:
                        sig_str = sig_str.split("->")[0]
                    tmp_locals = {}
                    exec("def " + sig_str + ": pass", {"torch": torch}, tmp_locals)
                    sig = inspect.signature(tmp_locals[attr])
                else:
                    sig = None
            except (AttributeError, SyntaxError):
                sig = None
    elif name == 'tf.nn':
        if not hasattr(tf.nn, attr):
            return None
        sig = getattr(tf.nn, attr)
        try:
            sig = inspect.signature(sig)
        except (ValueError, TypeError):
            sig = None
    elif name == 'tf':
        if not hasattr(tf, attr):
            return None
        sig = getattr(tf, attr)
        try:
            sig = inspect.signature(sig)
        except (ValueError, TypeError):
            sig = None
    else:
        raise NotImplementedError()

    if sig is None:
        if (name, attr) not in NO_SIGNATURES:
            # print("Signature not found:", name, attr)
            NO_SIGNATURES.add((name, attr))
        return None
    elif sig.parameters.keys() and not (set(sig.parameters.keys()) - {'args', 'kwargs'}):
        if (name, attr) not in VAR_SIGNATURES:
            # print("Signature for ", name, attr, " has parameters ", sig.parameters.keys())
            VAR_SIGNATURES.add((name, attr))
        return None

    return sig


def resolve_alias(attr: str) -> Optional[str]:
    if not hasattr(layers, attr):
        return None
    obj = getattr(layers, attr)
    if not hasattr(obj, "__name__"):
        return None
    if obj.__name__ != attr:
        if attr not in RESOLVED_ALIAS:
            RESOLVED_ALIAS.add(attr)
            print(f"layers.{attr} resolved to layers.{obj.__name__}")
        return obj.__name__
    else:
        return None
