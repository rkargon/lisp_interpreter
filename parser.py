import ast
import enum
import io
import tokenize as tknz
from typing import List, Optional, Generic, TypeVar, Any, Set

import attr


class TokenType(enum.Enum):
    OPEN_PAREN = enum.auto()
    CLOSE_PAREN = enum.auto()
    BOOL = enum.auto()
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    STRING = enum.auto()
    SYMBOL = enum.auto()
    BUILTIN = enum.auto()


_TT = TypeVar("_TT")


@attr.s
class Token(Generic[_TT]):
    type_: _TT = attr.ib()
    value: Any = attr.ib()


class Builtin:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value

    def __eq__(self, other):
        match other:
            case Builtin(v):
                return v == self.value
            case _:
                return False


class Symbol:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value


class Tokenizer(Generic[_TT]):
    def tokenize(self, string: str) -> List[Token[_TT]]:
        pass


class DefaultPythonTokenizer(Tokenizer[TokenType]):
    """
    Relies on Python's built-in tokenizer. As such, requires that tokens match Python's syntax.
    i.e. tokens can be strings, numbers, valid pthon builtin/variable names, etc.
    """

    def __init__(self, builtins: Set[Builtin]):
        self._builtins: Set[Builtin] = set(builtins)

    def tokenize(self, string: str) -> List[Token[TokenType]]:
        python_tokens = self._python_tokenize_string(string)
        tokens = list(filter(None, (self._match_python_token(tok) for tok in python_tokens)))
        return tokens

    def _match_python_token(self, token: tknz.TokenInfo) -> Optional[Token[TokenType]]:
        match (token.exact_type, token.string):
            case (tknz.LPAR, "("):
                return Token(TokenType.OPEN_PAREN, None)
            case (tknz.RPAR, ")"):
                return Token(TokenType.CLOSE_PAREN, None)
            case (tknz.NUMBER, number_string):
                try:
                    n = int(number_string)
                    return Token(TokenType.INTEGER, n)
                except ValueError:
                    n = float(number_string)
                    return Token(TokenType.FLOAT, n)
            case (tknz.NUMBER, float(n)):
                return Token(TokenType.FLOAT, n)
            case (tknz.STRING, s):
                return Token(TokenType.STRING, ast.literal_eval(s))
            case (tknz.NAME, "true" | "false" as s):
                return Token(TokenType.BOOL, bool(s == "true"))
            case (tknz.NAME, s) if Builtin(s) in self._builtins:
                return Token(TokenType.BUILTIN, Builtin(s))
            case (tknz.NAME, s):
                return Token(TokenType.SYMBOL, Symbol(s))
            case (t, op) if t in OPERATOR_SYMBOLS and Builtin(s) in self._builtins:
                return Token(TokenType.BUILTIN, Builtin(op))
            case _:
                return None

    def _python_tokenize_string(self, string: str) -> List[tknz.TokenInfo]:
        return list(tknz.tokenize(io.BytesIO(string.encode('utf-8')).readline))


OPERATOR_SYMBOLS = [
    tknz.PLUS,
    tknz.MINUS,
    tknz.STAR,
    tknz.SLASH,
    tknz.EQEQUAL,
    tknz.NOTEQUAL,
    tknz.EQUAL,
    tknz.LESS,
    tknz.GREATER,
    tknz.LESSEQUAL,
    tknz.GREATEREQUAL,
]
