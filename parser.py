import ast
import enum
import io
import tokenize as tknz
from abc import abstractmethod
from typing import List, Optional, TypeVar, Set, Iterable, Any

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
class Token:
    type_: TokenType = attr.ib()
    value: str = attr.ib()


@attr.s(auto_detect=True)
class Builtin:
    value: str = attr.ib()

    def __repr__(self):
        return self.value


@attr.s(auto_detect=True)
class Symbol:
    value: str = attr.ib()

    def __repr__(self):
        return self.value


class Tokenizer:
    def tokenize(self, string: str) -> List[Token]:
        pass


class DefaultPythonTokenizer(Tokenizer):
    """
    Relies on Python's built-in tokenizer. As such, requires that tokens match Python's syntax.
    i.e. tokens can be strings, numbers, valid python builtin/variable names, etc.
    """

    def __init__(self, builtins: Set[str]):
        self._builtins: Set[str] = set(builtins)

    def tokenize(self, string: str) -> List[Token]:
        python_tokens = self._python_tokenize_string(string)
        tokens = list(filter(None, (self._match_python_token(tok) for tok in python_tokens)))
        return tokens

    def _match_python_token(self, token: tknz.TokenInfo) -> Optional[Token]:
        match (token.exact_type, token.string):
            case (tknz.LPAR, "("):
                return Token(TokenType.OPEN_PAREN, "(")
            case (tknz.RPAR, ")"):
                return Token(TokenType.CLOSE_PAREN, "(")
            case (tknz.NUMBER, number_string):
                try:
                    n = int(number_string)
                    return Token(TokenType.INTEGER, number_string)
                except ValueError:
                    n = float(number_string)
                    return Token(TokenType.FLOAT, number_string)
            case (tknz.NUMBER, float(n)):
                return Token(TokenType.FLOAT, n)
            case (tknz.STRING, s):
                return Token(TokenType.STRING, ast.literal_eval(s))
            case (tknz.NAME, "true" | "false" as s):
                return Token(TokenType.BOOL, s)
            case (tknz.NAME, s) if s in self._builtins:
                return Token(TokenType.BUILTIN, s)
            case (tknz.NAME, s):
                return Token(TokenType.SYMBOL, s)
            case (_, op) if op in self._builtins:
                return Token(TokenType.BUILTIN, op)
            case _:
                return None

    def _python_tokenize_string(self, string: str) -> List[tknz.TokenInfo]:
        return list(tknz.tokenize(io.BytesIO(string.encode('utf-8')).readline))


@attr.s
class Parser:
    tokenizer: Tokenizer = attr.ib()

    def parse(self, s: str):
        tokens = self.tokenizer.tokenize(s)
        return self._parse_tokens(tokens)

    @abstractmethod
    def _parse_tokens(self, tokens: Iterable[Token]):
        raise NotImplementedError()


class LispParser(Parser):
    def __init__(self, builtins: Set[str]):
        self._builtins: Set[str] = builtins
        self.tokenizer: Tokenizer = DefaultPythonTokenizer(self._builtins)

    def _parse_tokens(self, tokens: Iterable[Token]):
        level = 0
        expressions: List[List[SExpr | Tuple]] = []

        for t in tokens:
            match t:
                case Token(type_=TokenType.OPEN_PAREN):
                    level += 1
                    expressions.append([])
                case Token(
                    type_=(TokenType.STRING | TokenType.FLOAT | TokenType.INTEGER | TokenType.BOOL | TokenType.SYMBOL | TokenType.BUILTIN),
                    value=v):
                    parsed_value = self._parse_value(t.type_, v)
                    if level == 0:
                        return parsed_value
                    expressions[-1].append(parsed_value)
                case Token(type_=TokenType.CLOSE_PAREN):
                    level -= 1
                    expr = tuple(expressions.pop())
                    if level == 0:
                        return expr
                    if level < 0:
                        raise RuntimeError("Mismatched parens!")
                    expressions[-1].append(expr)
                case _:
                    raise ValueError(f"Unexpected token: {t}")

    @classmethod
    def _parse_value(cls, type_: TokenType, value: str) -> Any:
        match type_:
            case TokenType.STRING:
                return value
            case TokenType.FLOAT:
                return float(value)
            case TokenType.INTEGER:
                return int(value)
            case TokenType.BOOL:
                return value == "true"
            case TokenType.SYMBOL:
                return Symbol(value)
            case TokenType.BUILTIN:
                return Builtin(value)
