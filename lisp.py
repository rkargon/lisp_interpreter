#!/usr/bin/env python3

import ast
import enum
import io
import pprint
import tokenize
import traceback
from typing import List, Iterable, Any, Optional, Tuple, Dict, TypeVar, Callable

import attr

from util import LinkedList, Link, Empty


@attr.s
class Dialect:
    parser = attr.ib()
    builtins = attr.ib()
    interpreter = attr.ib()


@attr.s
class Language(Dialect):
    army = attr.ib()
    navy = attr.ib()
    # hyuk hyuk





class Builtin(enum.Enum):
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

    # TODO: define other relational operations in terms of these

    @classmethod
    def is_builtin(cls, k: str) -> bool:
        try:
            Builtin(k)
            return True
        except ValueError:
            return False

    @classmethod
    def operators(cls) -> List["Builtin"]:
        return [
            Builtin.STRCAT,
            Builtin.PRINT,
            Builtin.ASSERT,
            #
            Builtin.ADD,
            Builtin.SUB,
            Builtin.MUL,
            Builtin.DIV,
            Builtin.EQ,
            Builtin.NEQ,
            Builtin.LT,
            Builtin.LTE,
            Builtin.GT,
            Builtin.GTE,
            #
            Builtin.LIST,
            Builtin.CONS,
            Builtin.HEAD,
            Builtin.TAIL,
        ]

    def __repr__(self):
        return self.value




Literal = int | float | str | bool
Atom = Literal | Symbol | Builtin
SExpr = Atom | LinkedList


@attr.s(auto_detect=True)
class Lambda:
    params: Tuple[Symbol, ...] = attr.ib()
    body: SExpr = attr.ib()
    scope: "Scope" = attr.ib()

    def eval(self, args: Iterable[SExpr], calling_scope: "Scope") -> "Value":
        lambda_scope = dict(self.scope)
        bound_params = {param.value: Reference.lazy(arg, calling_scope) for param, arg in zip(self.params, args)}
        lambda_scope.update(**bound_params)
        return eval_expr(self.body, scope=lambda_scope)

    def __repr__(self):
        return f"<{self.__class__.__name__.upper()} {print_expr(self.params)} => {print_expr(self.body)} scope={set(self.scope.keys())}>"


class Macro(Lambda):
    def eval(self, args: List[SExpr], calling_scope: "Scope") -> "Value":
        lambda_scope = dict(self.scope)
        bound_params = {param.value: Reference.value(arg) for param, arg in zip(self.params, args)}
        lambda_scope.update(**bound_params)
        resulting_expression = eval_expr(self.body, scope=lambda_scope)
        return eval_expr(resulting_expression, scope=calling_scope)


Value = Literal | Lambda | Builtin | LinkedList


def parse(tokens: List[Token]) -> SExpr | Tuple[SExpr, ...]:
    level = 0
    expressions: List[List[SExpr | Tuple]] = []

    for t in tokens:
        match t:
            case Token(type_=TokenType.OPEN_PAREN):
                level += 1
                expressions.append([])
            case Token(
                type_=TokenType.STRING | TokenType.FLOAT | TokenType.INTEGER | TokenType.BOOL | TokenType.SYMBOL | TokenType.BUILTIN,
                value=v):
                if level == 0:
                    return v
                expressions[-1].append(v)
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


def read_expr(s: str) -> SExpr:
    raw_tokens = tokenize_string(s)
    tokens = process_tokens(raw_tokens)
    expr_tuple = parse(tokens)
    return LinkedList.linkify(expr_tuple
                              )


class Reference:
    def __init__(self, expr: Optional[SExpr] = None, scope: Optional["Scope"] = None, value: Optional[Value] = None):
        self._expr = expr
        self._scope = scope
        self._value = value
        assert value is not None or expr is not None, "either value or expression must be present"

    def get(self) -> Value:
        if self._value:
            return self._value

        self._value = eval_expr(self._expr, self._scope)
        return self._value

    @classmethod
    def value(cls, value: Value) -> "Reference":
        return cls(value=value)

    @classmethod
    def lazy(cls, expr: SExpr, scope: Optional["Scope"]) -> "Reference":
        return cls(expr=expr, scope=scope)


Scope = Dict[str, Reference]


def eval_expr(expr: SExpr, scope: Optional[Scope] = None):
    # TODO type checking
    if scope is None:
        scope: Scope = {}
    match expr:
        case () | Empty() | Builtin.NIL:
            return Empty()
        case int() | float() | str() | bool() | Builtin():
            return expr
        case Symbol(value=v):
            return scope[v].get()
        case Link(value=Builtin.COND, rest=conditions):
            for test, action in conditions:
                test_result = eval_expr(test, scope)
                assert isinstance(test_result, bool), f"condition {print_expr(test)} must be bool"
                if test_result:
                    return eval_expr(action, scope)
            return Empty()
        case Link(value=Builtin.SET, rest=args):
            symbol, sub_expr = args
            new_value = eval_expr(sub_expr, scope)
            scope[symbol.value] = Reference.value(new_value)
            return new_value
        case Link(value=Builtin.LET, rest=args):
            name, value, body = args
            # TODO make this a macro / sugar
            new_expr = LinkedList.linkify(((Builtin.LAMBDA, (name,), body), value))
            return eval_expr(new_expr, scope)
        case Link(value=Builtin.LAMBDA, rest=args):
            params, body = args
            return Lambda(params=tuple(params), body=body, scope=scope)
        case Link(value=Builtin.MACRO, rest=args):
            params, body = args
            return Macro(params=tuple(params), body=body, scope=scope)
        case Link(value=Builtin.SEXPR, rest=body):
            return body
        case Link(value=Builtin.QUOTE, rest=Link(value=body, rest=Empty())):
            return body
        case Link(value=head, rest=args):
            head_eval = eval_expr(head, scope)
            match head_eval:
                case Builtin():
                    return apply_operator(head_eval, *[eval_expr(e, scope) for e in args])
                case Lambda():
                    return head_eval.eval(args, calling_scope=scope)
                case _:
                    raise ValueError(f"Head expression evaluated to non-callable: {print_expr(head_eval)}")
        case _:
            raise ValueError(f"Failed to evaluate expression: {print_expr(expr)}")


def apply_operator(op: Builtin, *args: Value) -> Value:
    match (op, *args):
        case (Builtin.ADD, *args):
            return sum(args)
        case (Builtin.SUB, first, second):
            return first - second
        case (Builtin.MUL, *args):
            return reduce(lambda x, y: x * y, args, initial=1)
        case (Builtin.DIV, dividend, divisor):
            return dividend / divisor
        case (Builtin.EQ, left, right):
            return left == right
        case (Builtin.LT, left, right):
            return left < right
        case (Builtin.LTE, left, right):
            return left <= right
        case (Builtin.GT, left, right):
            return left > right
        case (Builtin.GTE, left, right):
            return left >= right
        case (Builtin.READ, expr):
            return read_expr(expr)
        case (Builtin.EVAL, expr):
            return eval_expr(expr)
        case (Builtin.PRINT, arg):
            return print_expr(arg)
        case (Builtin.STRCAT, *args):
            return "".join(args)
        case (Builtin.STRFMT, s, *args):
            return s % tuple(args)
        case (Builtin.ASSERT, condition):
            assert condition
            return True
        case (Builtin.ASSERT, condition, message):
            assert condition, message
            return True
        case (Builtin.TYPEID, arg):
            return type(arg).__name__.lower()
        case (Builtin.SYMBOL, s):
            return isinstance(s, Symbol)
        # List ops
        case (Builtin.LIST, *args):
            return LinkedList.of(*args)
        case (Builtin.CONS, head, tail):
            return Link(head, tail)
        case (Builtin.HEAD | Builtin.TAIL, Empty()):
            raise ValueError(f"Can't call {op} on empty list")
        case (Builtin.HEAD, Link(head, _)):
            return head
        case (Builtin.TAIL, Link(_, tail)):
            return tail
        case _:
            raise ValueError(f"Failed to evaluate operator {print_expr((op, *args))}")


def print_expr(expr: SExpr | Tuple[SExpr, ...]) -> str:
    match expr:
        case int() | float() | bool():
            return str(expr).lower()
        case str():
            return repr(expr)
        case Symbol(value=v):
            return str(v)
        case Builtin():
            return repr(expr)
        case Lambda():
            return repr(expr)
        case tuple() | LinkedList():
            return "(" + " ".join(print_expr(e) for e in expr) + ")"
        case _:
            return repr(expr)


T = TypeVar("T")
U = TypeVar("U")


def reduce(fun: Callable[[U, T], U], items: List[T], initial: U) -> U:
    result = initial
    for i in items:
        result = fun(result, i)
    return result


PREAMBLES = {
    "builtins": """
        (set add (lambda (x y) (+ x y)))
        (set mul (lambda (x y) (* x y)))
        (set sub (lambda (x y) (- x y)))
        (set neg (lambda (x) (- 0 x)))
        (set div (lambda (x y) (/ x y)))
        (set eq (lambda (x y) (= x y)))
        (set printf (lambda (x) (print x))) 
    """,
    "bools": """
        (set if (lambda (condition if_true if_false) (cond (condition if_true) (true if_false))))
        (set and (lambda (a b) (if a b a)))
        (set or (lambda (a b) (if a a b)))
        (set not (lambda (a) (if a false true)))
    """,
    "pairs": """
        (set church_pair (lambda (x y) (lambda (z) (z x y))))
        (set church_first (lambda (l) (l (lambda (x y) x)))) 
        (set church_second (lambda (l) (l (lambda (x y) y))))
    """,
    "misc": """
    (set comp (lambda (f g) (lambda (x) (f (g x)))))
    (set fog comp)
    """,
    "list_ops": """
        (set is_nil (lambda (l) (= l nil)))
        (set fold (lambda (f l i) (if (is_nil l) i   (f (head l) (fold f (tail l) i)))))
        (set map (lambda (f l)   (if (is_nil l) nil (cons (f (head l)) (map f (tail l))))))
        (set map_val (lambda (i l) (map (lambda (_) i) l)))
        (set len (lambda (l) (fold add (map_val 1 l) 0)))
        (set cat (lambda (l1 l2) (if (is_nil l1) l2 (cons (head l1) (cat (tail l1) l2)))))
        (set range (lambda (start end) (and (assert (<= start end) (print (list start "<=" end))) (if (= start end) nil (cons start (range (+ start 1) end))))))
        (set flatten (lambda (l) (fold cat l nil)))
        (set cartesian_prod (lambda (l1 l2) (flatten (map (lambda (a) (map (lambda (b) (list a b)) l2)) l1))))
        (set first head)
        (set second (comp head tail))
    """,

}


def load_preamble(preamble: str, scope: Scope):
    print(f"Loading preamble...")
    lines = [s.strip() for s in preamble.strip().split("\n")]
    for l in lines:
        print(f"> {l}")
        expression = read_expr(l)
        eval_expr(expression, scope=scope)


def quick_eval(s: str) -> SExpr:
    return eval_expr(read_expr(s))


def main():
    # loads improved prompt interaction (e.g. arrow keys and the like)
    import readline
    readline.read_init_file("./editrc")

    scope: Scope = {}

    # load preambles
    # load_preamble(PREAMBLES["bools"], scope)
    for name in ["bools", "pairs", "builtins", "misc", "list_ops"]:
        preamble = PREAMBLES[name]
        print("Load preamble", name)
        load_preamble(preamble, scope)

    while True:
        string = input("> ")
        if not string:
            continue
        try:
            # parse special commands
            if string.startswith("!"):
                match string.split():
                    case "!preamble", name:
                        load_preamble(PREAMBLES[name], scope)
                    case ["!scope"]:
                        pprint.pprint(scope, indent=4)
                    case ["!scope", *expr]:
                        expr_string = " ".join(expr)
                        result = eval_expr(read_expr(expr_string), scope=scope)
                        if not isinstance(result, Lambda):
                            raise ValueError(f"Result [{result}] is not a lambda!")
                        pprint.pprint(result.scope, indent=4)
                    case _:
                        raise ValueError(f"Could not understand command {string}")
                continue
            else:
                expression = read_expr(string)
                result = eval_expr(expression, scope=scope)
                print(print_expr(result))
        except Exception as e:
            traceback.print_exception(e)


if __name__ == "__main__":
    main()
