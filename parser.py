import ast
import enum
import io
import tokenize as tknz
from abc import abstractmethod
from typing import List, Optional, TypeVar, Set, Iterable, Any, Tuple, Generic

import attr

from util import LinkedList


class TokenType(enum.Enum):
    OPEN_PAREN = enum.auto()
    CLOSE_PAREN = enum.auto()
    BOOL = enum.auto()
    INTEGER = enum.auto()
    FLOAT = enum.auto()
    STRING = enum.auto()
    SYMBOL = enum.auto()
    BUILTIN = enum.auto()
    COMMENT = enum.auto()


_TT = TypeVar("_TT")


@attr.s
class Token:
    type_: TokenType = attr.ib()
    value: str = attr.ib()


class Tokenizer:
    def tokenize(self, string: str) -> List[Token]:
        pass


@attr.s(auto_detect=True)
class Builtin:
    keyword: str = attr.ib()

    def __repr__(self):
        return str(self.keyword)

    def __eq__(self, other):
        if not isinstance(other, Builtin):
            return False
        return self.keyword == other.keyword

    def __hash__(self):
        return hash(self.keyword)



@attr.s(auto_detect=True)
class Symbol:
    value: str = attr.ib()

    def __repr__(self):
        return self.value


Literal = int | float | str | bool
Atom = Literal | Symbol | Builtin
# S-Expressions represented by a linked list (what's actually used by the interpreter at runtime
SExpr = Atom | LinkedList
# S-Expressions stored as recursive tuples, used in parsing
SExprTuple = Atom | Tuple


def python_tokenize_string(string: str) -> List[tknz.TokenInfo]:
    """Uses Python's builtin tokenization on a string"""
    return list(tknz.tokenize(io.BytesIO(string.encode('utf-8')).readline))


class DefaultPythonTokenizer(Tokenizer):
    """
    Relies on Python's built-in tokenizer. As such, requires that tokens match Python's syntax.
    i.e. tokens can be strings, numbers, valid python builtin/variable names, etc. '#' character will create a comment and rest of line will be ignored.
    We also interpret `;` as a comment for consistency w/ other LISPs
    """

    def __init__(self, builtins: Set[str]):
        self._builtins: Set[str] = set(builtins)

    def tokenize(self, string: str) -> List[Token]:
        python_tokens = python_tokenize_string(string)
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
                    _ = int(number_string)
                    return Token(TokenType.INTEGER, number_string)
                except ValueError:
                    _ = float(number_string)
                    return Token(TokenType.FLOAT, number_string)
            case (tknz.NUMBER, float(n)):
                return Token(TokenType.FLOAT, n)
            case (tknz.STRING, s):
                return Token(TokenType.STRING, ast.literal_eval(s))
            case (tknz.NAME, "true" | "false" as s) if s in self._builtins:
                return Token(TokenType.BOOL, s)
            case (tknz.NAME, s) if s in self._builtins:
                return Token(TokenType.BUILTIN, s)
            case (tknz.NAME, s):
                return Token(TokenType.SYMBOL, s)
            case (tknz.SEMI, ";"):
                return Token(TokenType.COMMENT, ";")
            case (_, op) if op in self._builtins:
                return Token(TokenType.BUILTIN, op)
            case _:
                return None


P = TypeVar("P")


@attr.s
class Parser(Generic[P]):

    @abstractmethod
    def parse(self, s: str) -> P:
        raise NotImplementedError()


class LispParser(Parser[SExpr]):
    def __init__(self, builtins: Set[str]):
        self._builtins: Set[str] = builtins
        self.tokenizer: Tokenizer = DefaultPythonTokenizer(self._builtins)

    def parse(self, s: str) -> SExpr:
        tokens = self.tokenizer.tokenize(s)
        tokens = self._preprocess(tokens)
        expr_tuple = self._parse_tokens(tokens)
        return LinkedList.linkify(expr_tuple)

    @classmethod
    def _preprocess(cls, tokens: List[Token]) -> List[Token]:
        comment_idx = len(tokens)
        for i, t in enumerate(tokens):
            if t.type_ == TokenType.COMMENT:
                comment_idx = i
                break
        return tokens[:comment_idx]

    @classmethod
    def _parse_tokens(cls, tokens: Iterable[Token]) -> SExprTuple:
        level = 0
        expressions: List[List[SExprTuple]] = []

        for t in tokens:
            match t:
                case Token(type_=TokenType.OPEN_PAREN):
                    level += 1
                    expressions.append([])
                case Token(
                    type_=(TokenType.STRING | TokenType.FLOAT | TokenType.INTEGER | TokenType.BOOL | TokenType.SYMBOL | TokenType.BUILTIN),
                    value=v):
                    parsed_value = cls._parse_value(t.type_, v)
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
                return {"true": True, "false": False}[value]
            case TokenType.SYMBOL:
                return Symbol(value)
            case TokenType.BUILTIN:
                return Builtin(value)


def print_sexpr(expr: SExpr) -> str:
    match expr:
        case int() | float() | bool():
            return str(expr).lower()
        case str():
            return repr(expr)
        case Symbol(value=v):
            return str(v)
        case Builtin():
            return repr(expr)
        case tuple() | LinkedList():
            return "(" + " ".join(print_sexpr(e) for e in expr) + ")"
        case _:
            return repr(expr)
