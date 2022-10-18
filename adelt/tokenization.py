import re
import sys
import tokenize
from io import BytesIO

import unicodedata
from sacrebleu import tokenize_v14_international

SPECIAL_OPEN = '\u3016'
SPECIAL_CLOSE = '\u3017'
SPECIAL_BLANK = '\u25af'

NON_PRINTING_CHAR_RE = re.compile('[%s]' % re.escape(
    ''.join(
        chr(i) for i in range(sys.maxunicode)
        if unicodedata.category(chr(i)) in {'Cc', 'Cf'}
    )
))
ESCAPE_SEQ_RE = re.compile(r'(\\(?:[\n\\\'"abfnrtv]|[0-7]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]))')


def escape(s: str):
    seq = re.split(f'([{SPECIAL_OPEN}{SPECIAL_CLOSE}])', s)
    translate = {
        SPECIAL_OPEN: SPECIAL_OPEN + "sopen" + SPECIAL_CLOSE,
        SPECIAL_CLOSE: SPECIAL_OPEN + "sclose" + SPECIAL_CLOSE,
        SPECIAL_BLANK: SPECIAL_OPEN + "sblank" + SPECIAL_CLOSE
    }
    seq = [
        translate.get(tok, tok)
        for tok in seq
    ]
    return "".join(seq)


def unescape(s: str):
    seq = re.split("(" + SPECIAL_OPEN + r"[a-z]+" + SPECIAL_CLOSE + ")", s)
    translate = {
        SPECIAL_OPEN + "sopen" + SPECIAL_CLOSE: SPECIAL_OPEN,
        SPECIAL_OPEN + "sclose" + SPECIAL_CLOSE: SPECIAL_CLOSE,
        SPECIAL_OPEN + "sblank" + SPECIAL_CLOSE: SPECIAL_BLANK
    }
    seq = [
        translate.get(tok, tok)
        for tok in seq
    ]
    return "".join(seq)


def process_string(s: str, revertible: bool = True, python_escape: bool = False) -> str:
    if not revertible:
        lines = []
        if python_escape:
            s = ''.join(s.split('\\\n'))  # line continuations
        for line in s.splitlines(keepends=False):
            line = NON_PRINTING_CHAR_RE.sub('', line)  # remove non-printing chars
            line = unicodedata.normalize("NFKD", line)  # NFKD normalize
            line = ' '.join(line.strip().split())  # normalize whitespace characters
            if python_escape:
                parts = []
                for part in re.split(ESCAPE_SEQ_RE, line):
                    if not part.startswith("\\"):
                        part = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", part)  # filter repetitive characters
                        part = tokenize_v14_international(part)  # add spaces before punctuations
                        part = tokenize_v14_international(part)
                    parts.append(part)
                line = ' '.join(parts)
            else:
                line = re.sub(r"(.)\1\1\1\1+", r"\1\1\1\1\1", line)  # filter repetitive characters
                line = tokenize_v14_international(line)  # add spaces before punctuations
                line = tokenize_v14_international(line)
            line = ' '.join(line.strip().split())
            if len(re.sub(r'\W', '', line)) >= 2:  # filter too short lines or no-words lines
                lines.append(line + '\n')
        if lines:
            lines[-1] = lines[-1].strip()
        s = ''.join(lines)
    s = escape(s)
    if revertible:
        s = s.replace("\n", SPECIAL_OPEN + "lf" + SPECIAL_CLOSE)
        s = s.replace("\r", SPECIAL_OPEN + "cr" + SPECIAL_CLOSE)
        s = s.replace("\t", SPECIAL_OPEN + "tab" + SPECIAL_CLOSE)
        s = s.replace(" ", SPECIAL_BLANK)
    else:
        s = s.replace("\n", " " + SPECIAL_OPEN + "lf" + SPECIAL_CLOSE + " ")
        s = ' '.join(s.strip().split())
    return s


def recover_string(s):
    s = s.replace(SPECIAL_BLANK, " ")
    s = s.replace(SPECIAL_OPEN + "tab" + SPECIAL_CLOSE, "\t")
    s = s.replace(SPECIAL_OPEN + "cr" + SPECIAL_CLOSE, "\r")
    s = s.replace(SPECIAL_OPEN + "lf" + SPECIAL_CLOSE, "\n")
    return unescape(s)


def parse_python_string_literal(s):
    modifier = ''
    while s and s[0] != '"' and s[0] != "'":
        modifier += s[0]
        s = s[1:]
    assert len(modifier) <= 2, "Invalid string modifier"
    modifier = modifier.lower()
    assert s, "Invalid empty string literal"
    if s.startswith('"""') and s.endswith('"""'):
        return '"""', modifier, s[3:-3]
    elif s.startswith("'''") and s.endswith("'''"):
        return "'''", modifier, s[3:-3]
    elif s.startswith('"') and s.endswith('"'):
        return '"', modifier, s[1:-1]
    elif s.startswith("'") and s.endswith("'"):
        return "'", modifier, s[1:-1]
    else:
        raise AssertionError("Invalid string delimiter")


class _TokenConverterPython:
    CONTINUE = 0
    EOS = 1
    ERROR = 2

    def __init__(self):
        self.tokens = []

    def step(self, toktype, tok, is_docstring) -> int:
        if toktype == tokenize.ENCODING or toktype == tokenize.NL:
            return self.CONTINUE

        elif toktype == tokenize.ENDMARKER:
            self.tokens.append(SPECIAL_OPEN + "endmarker" + SPECIAL_CLOSE)
            return self.EOS

        elif toktype == tokenize.INDENT:
            self.tokens.append(SPECIAL_OPEN + "indent" + SPECIAL_CLOSE)

        elif toktype == tokenize.DEDENT:
            # empty block
            if self.tokens and self.tokens[-1] == SPECIAL_OPEN + "indent" + SPECIAL_CLOSE:
                self.tokens = self.tokens[:-1]
            else:
                self.tokens.append(SPECIAL_OPEN + "dedent" + SPECIAL_CLOSE)

        elif toktype == tokenize.NEWLINE:
            if (
                self.tokens
                and self.tokens[-1] != SPECIAL_OPEN + "newline" + SPECIAL_CLOSE
                and self.tokens[-1] != SPECIAL_OPEN + "comclose" + SPECIAL_CLOSE
                and self.tokens[-1] != SPECIAL_OPEN + "docclose" + SPECIAL_CLOSE
            ):
                self.tokens.append(SPECIAL_OPEN + "newline" + SPECIAL_CLOSE)

        elif toktype == tokenize.COMMENT:
            assert tok and tok.startswith("#")
            com = process_string(tok[1:], revertible=False)
            if com:
                self.tokens.append(SPECIAL_OPEN + "comopen" + SPECIAL_CLOSE)
                self.tokens.extend(com.split())
                self.tokens.append(SPECIAL_OPEN + "comclose" + SPECIAL_CLOSE)

        elif toktype == tokenize.STRING:
            s_delim, s_mod, s_con = parse_python_string_literal(tok)
            if is_docstring:  # docstring
                doc = process_string(s_con, revertible=False, python_escape="r" not in s_mod)
                if doc:
                    self.tokens.append(SPECIAL_OPEN + "docopen" + s_mod + ":" + s_delim + SPECIAL_CLOSE)
                    self.tokens.extend(doc.split())
                    self.tokens.append(SPECIAL_OPEN + "docclose" + SPECIAL_CLOSE)
            else:  # ordinary strings
                s_con = process_string(s_con, revertible=True, python_escape="r" not in s_mod)
                assert ' ' not in s_con, f"s_con contains whitespaces {s_con}"
                self.tokens.append(SPECIAL_OPEN + "stropen" + s_mod + ":" + s_delim + SPECIAL_CLOSE)
                if s_con:
                    self.tokens.append(s_con)
                self.tokens.append(SPECIAL_OPEN + "strclose" + SPECIAL_CLOSE)

        elif toktype == tokenize.ERRORTOKEN:
            return self.ERROR

        else:
            assert ' ' not in tok, f"tok contains whitespaces {tok}"
            self.tokens.append(tok)

        return self.CONTINUE

    def get_results(self):
        assert (self.tokens[-1] == SPECIAL_OPEN + "endmarker" + SPECIAL_CLOSE), "Error, no end marker"
        return self.tokens[:-1]


def tokenize_python(s: str) -> list:
    try:
        assert isinstance(s, str)
        s = s.replace("\r", "")
        try:
            iterator = tokenize.tokenize(BytesIO(s.encode('utf-8')).readline)
        except SyntaxError as excep:
            raise SyntaxError(excep)
        try:
            converter = _TokenConverterPython()
            for toktype, tok, _, _, line in iterator:
                is_docstring = toktype == tokenize.STRING and tok == line.rsplit('#', 1)[0].strip()
                status = converter.step(toktype, tok, is_docstring)
                if status == _TokenConverterPython.EOS:
                    break
                elif status == _TokenConverterPython.ERROR:
                    return []
                else:
                    assert status == _TokenConverterPython.CONTINUE
            return converter.get_results()
        except (tokenize.TokenError, IndentationError, SyntaxError, UnicodeDecodeError):
            raise Exception(
                f"Impossible to parse tokens because icorrect source code ...")
        except StopIteration:
            raise Exception(f"End of iterator before ENDMARKER token.")
    except KeyboardInterrupt:
        raise
    except AssertionError as e:
        print(e)
        return []
    except:
        return []


def detokenize_python(tokens: list) -> str:
    PYTHON_SPECIAL_TOKENS = {
        SPECIAL_OPEN + "indent" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "dedent" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "comopen" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "comclose" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "docclose" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "strclose" + SPECIAL_CLOSE,
        SPECIAL_OPEN + "newline" + SPECIAL_CLOSE
    }
    try:
        assert isinstance(tokens, list)
        indents = 0
        startline = False
        status = "code"
        code_tokens = []
        com_tokens = None
        doc_tokens = None
        str_tokens = None
        s_mod, s_delim = None, None
        for tok in tokens:
            if status == "code":
                assert com_tokens is None and doc_tokens is None and str_tokens is None
                if tok == SPECIAL_OPEN + "indent" + SPECIAL_CLOSE:
                    indents += 1
                elif tok == SPECIAL_OPEN + "dedent" + SPECIAL_CLOSE:
                    indents -= 1
                elif tok == SPECIAL_OPEN + "comopen" + SPECIAL_CLOSE:
                    status = "com"
                    com_tokens = []
                elif tok.startswith(SPECIAL_OPEN + "docopen"):
                    s_mod, s_delim = tok[len(SPECIAL_OPEN + "docopen"):-len(SPECIAL_CLOSE)].split(":")
                    status = "doc"
                    doc_tokens = []
                elif tok.startswith(SPECIAL_OPEN + "stropen"):
                    s_mod, s_delim = tok[len(SPECIAL_OPEN + "stropen"):-len(SPECIAL_CLOSE)].split(":")
                    status = "str"
                    str_tokens = []
                elif tok == SPECIAL_OPEN + "newline" + SPECIAL_CLOSE:
                    startline = True
                else:
                    assert tok not in PYTHON_SPECIAL_TOKENS
                    if startline:
                        code_tokens.append("\n")
                        code_tokens.append('    ' * indents)
                        startline = False
                    elif code_tokens:
                        code_tokens.append(' ')
                    code_tokens.append(tok)
            elif status == "com":
                assert com_tokens is not None
                if tok == SPECIAL_OPEN + "comclose" + SPECIAL_CLOSE:
                    if startline:
                        code_tokens.append("\n")
                        code_tokens.append('    ' * indents)
                    elif code_tokens:
                        code_tokens.append(' ')
                    code_tokens.append("# " + recover_string(" ".join(com_tokens)))
                    com_tokens = None
                    status = "code"
                    startline = True
                else:
                    assert tok not in PYTHON_SPECIAL_TOKENS
                    com_tokens.append(tok)
            elif status == "doc":
                assert doc_tokens is not None
                if tok == SPECIAL_OPEN + "docclose" + SPECIAL_CLOSE:
                    if startline:
                        code_tokens.append("\n")
                        code_tokens.append('    ' * indents)
                    elif code_tokens:
                        code_tokens.append(' ')
                    doc_str = recover_string(" ".join(doc_tokens))
                    if doc_str and doc_str[-1] == s_delim[0]:
                        doc_str = doc_str + " "
                    code_tokens.append(s_mod + s_delim + doc_str + s_delim)
                    doc_tokens = None
                    status = "code"
                    startline = True
                else:
                    assert tok not in PYTHON_SPECIAL_TOKENS
                    doc_tokens.append(tok)
            elif status == "str":
                assert str_tokens is not None
                if tok == SPECIAL_OPEN + "strclose" + SPECIAL_CLOSE:
                    if startline:
                        code_tokens.append("\n")
                        code_tokens.append('    ' * indents)
                        startline = False
                    elif code_tokens:
                        code_tokens.append(' ')
                    code_tokens.append(s_mod + s_delim + recover_string("".join(str_tokens)) + s_delim)
                    str_tokens = None
                    status = "code"
                else:
                    assert tok not in PYTHON_SPECIAL_TOKENS
                    str_tokens.append(tok)
        untok_s = "".join(code_tokens)
        return untok_s
    except KeyboardInterrupt:
        raise
    # except AssertionError as e:
    #     print(e)
    #     return ""
    # except:
    #     return ""


class _PythonTextScopeTracker:
    """
    Keep track of free text scopes (comments or docstrings)
    """

    def __init__(self):
        self._dep_com = 0
        self._dep_doc = 0

    def step(self, token):
        """
        Return whether the current token is a special control symbol
        """
        if token.startswith(SPECIAL_OPEN + "comopen"):
            self._dep_com += 1
        elif token.startswith(SPECIAL_OPEN + "comclose"):
            self._dep_com -= 1
            assert self._dep_com >= 0
        elif token.startswith(SPECIAL_OPEN + "docopen"):
            self._dep_doc += 1
        elif token.startswith(SPECIAL_OPEN + "docclose"):
            self._dep_doc -= 1
            assert self._dep_doc >= 0
        else:
            return False
        return True

    def is_text(self):
        return self._dep_com > 0 or self._dep_doc > 0


def python_2to3(tokens):
    """
    fast python2 to python3 without AST parsing, can be incomplete
    """
    is_print = False
    is_raise = False
    res = []
    text_scope = _PythonTextScopeTracker()
    for token in tokens:
        text_scope.step(token)

        res.append(token)

        if text_scope.is_text():
            continue

        if len(res) >= 2 and res[-2] == "print" and res[-1] != "(":
            # print ... -> print(...)
            if len(res) < 3 or res[-3] != ".":
                # ignore methods
                t = res.pop()
                res.append("(")
                res.append(t)
                is_print = True
        elif len(res) >= 2 and tuple(res[-2:]) == ("raw_input", "("):
            # raw_input ( -> input (
            if len(res) < 3 or res[-3] != ".":
                # ignore methods
                res[-2] = "input"
        elif len(res) >= 2 and tuple(res[-2:]) == ("xrange", "("):
            # xrange ( -> range (
            if len(res) < 3 or res[-3] != ".":
                # ignore methods
                res[-2] = "range"
        elif len(res) >= 4 and tuple(res[-4:]) == (".", "iterkeys", "(", ")"):
            # . iterkeys ( ) -> . keys ( )
            res[-3] = "keys"
        elif len(res) >= 4 and tuple(res[-4:]) == (".", "itervalues", "(", ")"):
            # . itervalues ( ) -> . values ( )
            res[-3] = "values"
        elif len(res) >= 4 and tuple(res[-4:]) == (".", "iteritems", "(", ")"):
            # . iteritems ( ) -> . items ( )
            res[-3] = "items"
        elif len(res) >= 3 and res[-3] == "raise" and res[-2].isidentifier() and res[-1] == ",":
            # raise xX , ... ->  # raise xX ( ... )
            res[-1] = "("
            is_raise = True
        elif len(res) >= 4 and res[-4] == "except" and res[-3].isidentifier() and res[-2] == "," and res[
            -1].isidentifier():
            # except xX , xX -> except xX as xX
            res[-2] = "as"
        elif len(res) >= 2 and res[-2].isnumeric() and res[-1] == "L":
            # 123L -> 123
            res.pop()
        elif len(res) >= 4 and tuple(res[-4:]) == (".", "xreadlines", "(", ")"):
            # . xreadlines ( ) ->
            res = res[:-4]

        if is_print and token == SPECIAL_OPEN + "newline" + SPECIAL_CLOSE:
            t = res.pop()
            res.append(")")
            res.append(t)
            is_print = False
        if is_raise and token == SPECIAL_OPEN + "newline" + SPECIAL_CLOSE:
            t = res.pop()
            res.append(")")
            res.append(t)
            is_raise = False

    return res


def extract_functions_python(tokens):
    try:
        functions_standalone = []
        functions_class = []
        status = 0
        function = []
        n_indent = 0
        text_scope = _PythonTextScopeTracker()
        for token in tokens:
            text_scope.step(token)
            if status == 0:
                if token == "def" and not text_scope.is_text():
                    function = ["def"]
                    status = 1
            elif status == 1:
                if token.isidentifier():
                    function.append(token)
                    status = 2
                else:
                    status = 0
            elif status == 2:
                function.append(token)
                if token == SPECIAL_OPEN + "indent" + SPECIAL_CLOSE:
                    n_indent += 1
                elif token == SPECIAL_OPEN + "dedent" + SPECIAL_CLOSE:
                    n_indent -= 1
                    if n_indent == 0:
                        status = 0
                        if function[function.index('(') + 1] == "self":
                            functions_class.append(python_2to3(function))
                        else:
                            functions_standalone.append(python_2to3(function))
        return functions_standalone, functions_class
    except KeyboardInterrupt:
        raise
    except AssertionError as e:
        print(e)
        return [], []
    except:
        return [], []


def remove_comments_and_docs(tokens):
    ret = []
    text_scope = _PythonTextScopeTracker()
    for token in tokens:
        if not text_scope.step(token):
            if not text_scope.is_text():
                ret.append(token)
    return ret


def extract_dl_python(tokens):
    try:
        pytorch_classes = []
        keras_classes = []
        status = 0
        function = []
        n_indent = 0
        text_scope = _PythonTextScopeTracker()
        for token in tokens:
            text_scope.step(token)
            if status == 0:
                if token == "class" and not text_scope.is_text():
                    function = ["class"]
                    status = 1
            elif status == 1:
                if token.isidentifier():
                    function.append(token)
                    status = 2
                else:
                    status = 0
            elif status == 2:
                function.append(token)
                if token == SPECIAL_OPEN + "indent" + SPECIAL_CLOSE:
                    n_indent += 1
                elif token == SPECIAL_OPEN + "dedent" + SPECIAL_CLOSE:
                    n_indent -= 1
                    if n_indent == 0:
                        status = 0
                        if "(" not in function:
                            continue
                        pi = function.index('(')
                        if (
                            (function[pi + 1: pi + 4] == ['nn', '.', 'Module'])
                            or (function[pi + 1: pi + 6] == ['torch', '.', 'nn', '.', 'Module'])
                        ):
                            pytorch_classes.append(python_2to3(function))
                        elif (
                            (function[pi + 1: pi + 4] == ['keras', '.', 'Model'])
                            or (function[pi + 1: pi + 6] == ['tf', '.', 'keras', '.', 'Model'])
                            or (function[pi + 1: pi + 4] == ['layers', '.', 'Layer'])
                            or (function[pi + 1: pi + 6] == ['keras', '.', 'layers', '.', 'Layer'])
                            or (function[pi + 1: pi + 8] == ['tf', '.', 'keras', '.', 'layers', '.', 'Layer'])
                        ):
                            keras_classes.append(python_2to3(function))
        return pytorch_classes, keras_classes
    except KeyboardInterrupt:
        raise
    # except AssertionError as e:
    #     print(e)
    #     return [], []
    # except:
    #     return [], []
