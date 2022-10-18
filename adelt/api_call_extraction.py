import ast
from dataclasses import dataclass
from typing import List

from .dl_module_utils import get_signature

N_KINDS = 3


@dataclass
class ApiCallInfo:
    KIND_LAYER = 0
    KIND_FUNC = 1
    KIND_KEYWORD = 2  # auxiliary

    kind: int
    span: int
    api_name: int
    arg_kws: list


class ApiCallExtractor(ast.NodeVisitor):
    def __init__(self, lang):
        super().__init__()
        self.lang = lang
        self.result = {}
        self.span_cnt = 0
        self.api_calls: List[ApiCallInfo] = []

    def _add_call_info(self, node, kind, sig):
        try:
            bind = sig.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords if kw.arg is not None})
        except TypeError:
            return
        self.result[node] = f"span_{self.span_cnt}"
        span = self.span_cnt
        self.span_cnt += 1
        self.result[node.func] = f"span_{self.span_cnt}"
        api_name = self.span_cnt
        self.span_cnt += 1
        arg_kws = []
        for keyword in node.keywords:
            if keyword.arg and keyword.arg in bind.arguments.keys():
                self.result[keyword] = f"keyword:span_{self.span_cnt}"
                arg_kws.append(self.span_cnt)
                self.span_cnt += 1
        self.api_calls.append(ApiCallInfo(kind, span, api_name, arg_kws))

    def visit_Call(self, node):
        super().generic_visit(node)
        # nn.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'nn'
            and self.lang == 'pytorch'
        ):
            sig = get_signature('nn', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_LAYER, sig)
        # layers.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'layers'
            and self.lang == 'keras'
        ):
            sig = get_signature('layers', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_LAYER, sig)
        # F.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'F'
            and self.lang == 'pytorch'
        ):
            sig = get_signature('F', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_FUNC, sig)
        # torch.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'torch'
            and self.lang == 'pytorch'
        ):
            sig = get_signature('torch', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_FUNC, sig)
        # tf.nn.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'tf'
            and node.func.value.attr == 'nn'
            and self.lang == 'keras'
        ):
            sig = get_signature('tf.nn', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_FUNC, sig)
        # tf.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'tf'
            and self.lang == 'keras'
        ):
            sig = get_signature('tf', node.func.attr)
            if sig is not None:
                self._add_call_info(node, ApiCallInfo.KIND_FUNC, sig)

    def build_node_to_span(self):
        def node_to_span(node):
            r = self.result.get(node, None)
            return [r] if r is not None else []

        return node_to_span
