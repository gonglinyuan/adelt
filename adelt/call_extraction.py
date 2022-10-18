import ast
from typing import List


class CallExtractor(ast.NodeVisitor):
    def __init__(self, lang):
        super().__init__()
        self.lang = lang
        self.result = {}
        self.span_cnt = 0
        self.calls: List[int] = []

    def _add_call_info(self, node):
        self.result[node] = f"span_{self.span_cnt}"
        span = self.span_cnt
        self.span_cnt += 1
        self.calls.append(span)

    def visit_Call(self, node):
        super().generic_visit(node)
        self._add_call_info(node)

    def build_node_to_span(self):
        def node_to_span(node):
            r = self.result.get(node, None)
            return [r] if r is not None else []

        return node_to_span
