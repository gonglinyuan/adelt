import ast
import inspect

from .dl_module_utils import get_signature, resolve_alias


class ModuleCanonicalizer(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        # tf.keras.xxx -> keras.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'tf'
            and node.value.attr == 'keras'
        ):
            return ast.Attribute(value=ast.Name(id='keras'), attr=node.attr)
        # keras.layers.xxx -> layers.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'keras'
            and node.value.attr == 'layers'
        ):
            return ast.Attribute(value=ast.Name(id='layers'), attr=node.attr)
        # tf.keras.layers.xxx -> layers.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == 'tf'
            and node.value.value.attr == 'keras'
            and node.value.attr == 'layers'
        ):
            return ast.Attribute(value=ast.Name(id='layers'), attr=node.attr)
        # torch.nn.xxx -> nn.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'torch'
            and node.value.attr == 'nn'
        ):
            return ast.Attribute(value=ast.Name(id='nn'), attr=node.attr)
        # nn.functional.xxx -> F.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == 'nn'
            and node.value.attr == 'functional'
        ):
            return ast.Attribute(value=ast.Name(id='F'), attr=node.attr)
        # torch.nn.functional.xxx -> F.xxx
        if (
            isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Attribute)
            and isinstance(node.value.value.value, ast.Name)
            and node.value.value.value.id == 'torch'
            and node.value.value.attr == 'nn'
            and node.value.attr == 'functional'
        ):
            return ast.Attribute(value=ast.Name(id='F'), attr=node.attr)
        return node


class AliasCanonicalizer(ast.NodeTransformer):
    def visit_Attribute(self, node: ast.Attribute):
        # layers.XXX
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == 'layers'
        ):
            orig_name = resolve_alias(node.attr)
            if orig_name is not None:
                return ast.Attribute(value=node.value, attr=orig_name)
        return node


class ApiCallCanonicalizer(ast.NodeTransformer):
    def visit_Call(self, node):
        sig = None
        # nn.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'nn'
        ):
            sig = get_signature('nn', node.func.attr)
        # F.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'F'
        ):
            sig = get_signature('F', node.func.attr)
        # layers.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'layers'
        ):
            sig = get_signature('layers', node.func.attr)
        # torch.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'torch'
        ):
            sig = get_signature('torch', node.func.attr)
        # tf.nn.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == 'tf'
            and node.func.value.attr == 'nn'
        ):
            sig = get_signature('tf.nn', node.func.attr)
        # tf.XXX
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == 'tf'
        ):
            sig = get_signature('tf', node.func.attr)

        if sig is not None:
            has_normal_args = False
            var_positionals = set()
            var_keywords = set()
            for param in sig.parameters.values():
                if param.kind == inspect.Parameter.VAR_POSITIONAL:
                    var_positionals.add(param.name)
                elif param.kind == inspect.Parameter.VAR_KEYWORD:
                    var_keywords.add(param.name)
                else:
                    has_normal_args = True
            if not has_normal_args:
                return node

            extra_kwargs = []
            for arg in node.args:
                if isinstance(arg, ast.Starred):
                    return node
            for kw in node.keywords:
                if kw.arg is None:
                    extra_kwargs.append(kw.value)

            try:
                bind = sig.bind(*node.args, **{kw.arg: kw.value for kw in node.keywords if kw.arg is not None})
            except TypeError:
                return node

            def _add(vv):
                if isinstance(vv, ast.expr):
                    return vv
                elif isinstance(vv, tuple):
                    return ast.Tuple(elts=[_add(it) for it in vv], ctx=ast.Load())
                elif isinstance(vv, dict):
                    return ast.Dict(
                        keys=[_add(it) for it in vv.keys()],
                        values=[_add(it) for it in vv.values()]
                    )
                elif isinstance(vv, str):
                    return ast.Constant(value=vv, kind=None)
                else:
                    raise ValueError()

            ast_bind = [
                ast.keyword(k, _add(v))
                for k, v in bind.arguments.items()
                if k not in var_positionals and k not in var_keywords
            ]
            ast_bind.extend([
                ast.keyword(kk, _add(vv))
                for k, v in bind.arguments.items()
                if k in var_keywords
                for kk, vv in v.items()
            ])
            ast_bind.extend([ast.keyword(None, _add(v)) for v in extra_kwargs])
            return ast.Call(func=node.func, args=[], keywords=ast_bind)
        return node
