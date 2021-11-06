#!/usr/bin/env python3

import enum
from abc import abstractmethod
from typing import List, Iterable, Optional, Tuple, Dict, Callable, Set

import attr

import util
from parser import SExpr, Symbol, Builtin, print_sexpr, LispParser
from util import LinkedList, Link, Empty


Scope = Dict[str, "Reference"]


class Interpreter:

    @abstractmethod
    def init(self):
        raise NotImplementedError()

    @abstractmethod
    def read(self, string: str) -> Optional[SExpr]:
        raise NotImplementedError()

    @abstractmethod
    def eval(self, expr: SExpr) -> SExpr:
        raise NotImplementedError()

    @abstractmethod
    def print(self, expr: SExpr) -> str:
        raise NotImplementedError()

    def repl(self, string: str) -> Optional[str]:
        expr = self.read(string)
        if expr is None:
            return None
        return self.print(self.eval(expr))

    @classmethod
    def quick_eval(cls, string: str, **kwargs):
        interpreter = cls(**kwargs)
        interpreter.init()
        return interpreter.repl(string)

    def load_file(self, filename: str):
        with open(filename, "r") as f:
            for l in f.readlines():
                lstrip = l.strip()
                if not lstrip:
                    continue
                print("> ", lstrip)
                self.repl(lstrip)


class Builtins(Builtin, enum.Enum):
    LET = "let"
    SET = "set"
    LAMBDA = "lambda"
    MACRO = "macro"
    COND = "cond"

    # List ops
    NIL = "nil"
    LIST = "list"
    HEAD = "head"
    TAIL = "tail"
    CONS = "cons"
    # sexpr
    SEXPR = "sexpr"
    QUOTE = "quote"
    SYMBOL = "symbol"

    READ = "read"
    EVAL = "eval"
    PRINT = "print"

    TYPEID = "typeid"
    STRCAT = "strcat"
    STRFMT = "strfmt"
    ASSERT = "assert"

    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    EQ = "="
    NEQ = "!="
    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="


class LispInterpreter(Interpreter):

    def __init__(self, builtins: Optional[Set[str]] = None):
        if builtins is None:
            builtins = set(x.keyword for x in Builtins)
        self.parser = LispParser(builtins)
        self.scope: Scope = {}

    def init(self):
        self.scope = {}

    def read(self, string: str) -> SExpr:
        return self.parser.parse(string)

    def eval(self, expr: SExpr) -> SExpr:
        return self._eval_with_scope(expr, self.scope)

    def _eval_with_scope(self, expr: SExpr, scope: Scope):
        match expr:
            case () | Empty() | Builtins.NIL:
                return Empty()
            case int() | float() | str() | bool() | Builtin():
                return expr
            case Symbol(value=v):
                return scope[v].get()
            case Link(value=Builtins.COND, rest=conditions):
                for test, action in conditions:
                    test_result = self._eval_with_scope(test, scope)
                    assert isinstance(test_result, bool), f"condition {self.print(test)} must be bool"
                    if test_result:
                        return self._eval_with_scope(action, scope)
                return Empty()
            case Link(value=Builtins.SET, rest=args):
                symbol, sub_expr = args
                new_value = self._eval_with_scope(sub_expr, scope)
                scope[symbol.value] = Reference.value(new_value)
                return new_value
            case Link(value=Builtins.LET, rest=args):
                name, value, body = args
                # TODO make this a macro / sugar
                new_expr = LinkedList.linkify(((Builtins.LAMBDA, (name,), body), value))
                return self._eval_with_scope(new_expr, scope)
            case Link(value=Builtins.LAMBDA, rest=args):
                params, body = args
                return Lambda(params=tuple(params), body=body, scope=scope)
            case Link(value=Builtins.MACRO, rest=args):
                params, body = args
                return Macro(params=tuple(params), body=body, scope=scope)
            case Link(value=Builtins.SEXPR, rest=body):
                return body
            case Link(value=Builtins.QUOTE, rest=Link(value=body, rest=Empty())):
                return body
            case Link(value=head, rest=args):
                head_eval = self._eval_with_scope(head, scope)
                match head_eval:
                    case Builtin():
                        return self._apply_operator(head_eval, *[self._eval_with_scope(e, scope) for e in args])
                    case Macro():
                        lambda_scope = dict(head_eval.scope)
                        bound_params = {param.value: Reference.value(arg) for param, arg in zip(head_eval.params, args)}
                        lambda_scope.update(**bound_params)
                        resulting_expression = self._eval_with_scope(head_eval.body, scope=lambda_scope)
                        return self._eval_with_scope(resulting_expression, scope=scope)
                    case Lambda():
                        lambda_scope = dict(head_eval.scope)
                        bound_params = {param.value: Reference.lazy(lambda a=arg: self._eval_with_scope(a, scope)) for
                                        param, arg in zip(head_eval.params, args)}
                        lambda_scope.update(**bound_params)
                        return self._eval_with_scope(head_eval.body, scope=lambda_scope)
                    case _:
                        raise ValueError(f"Head expression evaluated to non-callable: {self.print(head_eval)}")
            case _:
                raise ValueError(f"Failed to evaluate expression: {self.print(expr)}")

    def _apply_operator(self, op: Builtin, *args: SExpr) -> SExpr:
        match (op, *args):
            case (Builtins.ADD, *args):
                return sum(args)
            case (Builtins.SUB, first, second):
                return first - second
            case (Builtins.MUL, *args):
                return util.reduce(lambda x, y: x * y, args, initial=1)
            case (Builtins.DIV, dividend, divisor):
                return dividend / divisor
            case (Builtins.EQ, left, right):
                return left == right
            case (Builtins.LT, left, right):
                return left < right
            case (Builtins.LTE, left, right):
                return left <= right
            case (Builtins.GT, left, right):
                return left > right
            case (Builtins.GTE, left, right):
                return left >= right
            case (Builtins.READ, expr):
                return self.read(expr)
            case (Builtins.EVAL, expr):
                return self.eval(expr)
            case (Builtins.PRINT, arg):
                return self.print(arg)
            case (Builtins.STRCAT, *args):
                return "".join(args)
            case (Builtins.STRFMT, s, *args):
                return s % tuple(args)
            case (Builtins.ASSERT, condition):
                assert condition
                return True
            case (Builtins.ASSERT, condition, message):
                assert condition, message
                return True
            case (Builtins.TYPEID, arg):
                return type(arg).__name__.lower()
            case (Builtins.SYMBOL, s):
                return isinstance(s, Symbol)
            # List ops
            case (Builtins.LIST, *args):
                return LinkedList.of(*args)
            case (Builtins.CONS, head, tail):
                return Link(head, tail)
            case (Builtins.HEAD | Builtins.TAIL, Empty()):
                raise ValueError(f"Can't call {op} on empty list")
            case (Builtins.HEAD, Link(head, _)):
                return head
            case (Builtins.TAIL, Link(_, tail)):
                return tail
            case _:
                raise ValueError(f"Failed to evaluate operator {self.print((op, *args))}")

    def print(self, expr: SExpr) -> str:
        return print_sexpr(expr)


@attr.s(auto_detect=True)
class Lambda:
    params: Tuple[Symbol, ...] = attr.ib()
    body: SExpr = attr.ib()
    scope: "Scope" = attr.ib()

    def __repr__(self):
        return f"<{self.__class__.__name__.upper()} {print_sexpr(self.params)} => {print_sexpr(self.body)}>"


class Macro(Lambda):
    pass


class Reference:
    def __init__(self, thunk: Optional[Callable[[], SExpr]] = None, value: Optional[SExpr] = None):
        self._thunk = thunk
        self._value = value
        assert value is not None or thunk is not None, "either value or thunk must be present"

    def get(self) -> SExpr:
        if self._value is not None:
            return self._value

        self._value = self._thunk()
        return self._value

    @classmethod
    def value(cls, value: SExpr) -> "Reference":
        return cls(value=value)

    @classmethod
    def lazy(cls, thunk: Callable[[], SExpr]) -> "Reference":
        return cls(thunk=thunk)

    def __repr__(self):
        val_str = str(self._value) if self._value is not None else "___"
        thunk_str = f"({self._thunk})" if self._value is None else ""
        return f"<ref: {val_str} {thunk_str}>"

