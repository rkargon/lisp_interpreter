#!/usr/bin/env python3

# TODO parser-combinators?
import ast
import enum
import io
import pprint
import tokenize
import traceback
from typing import List, Generator, Iterable, Any, Optional, Tuple, Dict, TypeVar, Callable

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


# atom: int|float|str|symbol|builtin
# expr: atom | `(` expr* `)`


class Builtin(enum.Enum):
    LET = "let"
    LAMBDA = "lambda"
    IF = "if"
    # AND = "and"
    # OR = "or"
    # NOT = "not"

    # List ops
    NIL = "nil"
    LIST = "list"
    HEAD = "head"
    TAIL = "tail"
    CONS = "cons"
    # sexpr
    SEXPR = "sexpr"

    STRCAT = "strcat"
    PRINT = "print"
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


@attr.s
class Token:
    type_: TokenType = attr.ib()
    value: Any = attr.ib()


class Symbol:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return self.value


Literal = int | float | str | bool
Atom = Literal | Symbol | Builtin
SExpr = Atom | Tuple[Atom, ...]


@attr.s(auto_detect=True)
class Lambda:
    params: Tuple[Symbol, ...] = attr.ib()
    body: SExpr = attr.ib()
    scope: Dict[str, "Value"] = attr.ib()

    def __repr__(self):
        return f"<LAMBDA {print_expr(self.params)} => {print_expr(self.body)} scope={set(self.scope.keys())}>"


@attr.s(auto_detect=True)
class ListValue:
    items: Tuple["Value", ...] = attr.ib()

    def __repr__(self):
        return "[" + " ".join(map(print_expr, self.items)) + "]"


Value = Literal | Lambda | ListValue | Builtin


def process_tokens(raw_tokens: List[tokenize.TokenInfo]) -> List[Token]:
    return list(filter(None, map(match_token, raw_tokens)))


OPERATOR_SYMBOLS = [
    tokenize.PLUS,
    tokenize.MINUS,
    tokenize.STAR,
    tokenize.SLASH,
    tokenize.EQEQUAL,
    tokenize.NOTEQUAL,
    tokenize.EQUAL,
    tokenize.LESS,
    tokenize.GREATER,
    tokenize.LESSEQUAL,
    tokenize.GREATEREQUAL,

]


def match_token(token: tokenize.TokenInfo) -> Optional[Token]:
    match (token.exact_type, token.string):
        case (tokenize.LPAR, "("):
            return Token(TokenType.OPEN_PAREN, None)
        case (tokenize.RPAR, ")"):
            return Token(TokenType.CLOSE_PAREN, None)
        case (tokenize.NUMBER, number_string):
            try:
                n = int(number_string)
                return Token(TokenType.INTEGER, n)
            except ValueError:
                n = float(number_string)
                return Token(TokenType.FLOAT, n)
        case (tokenize.NUMBER, float(n)):
            return Token(TokenType.FLOAT, n)
        case (tokenize.STRING, s):
            return Token(TokenType.STRING, ast.literal_eval(s))
        case (tokenize.NAME, "true" | "false" as s):
            return Token(TokenType.BOOL, bool(s == "true"))
        case (tokenize.NAME, s) if Builtin.is_builtin(s):
            return Token(TokenType.BUILTIN, Builtin(s))
        case (tokenize.NAME, s):
            return Token(TokenType.SYMBOL, Symbol(s))
        case (t, op) if t in OPERATOR_SYMBOLS and Builtin.is_builtin(op):
            return Token(TokenType.BUILTIN, Builtin(op))
        case _:
            return None


def tokenize_string(s: str) -> List[tokenize.TokenInfo]:
    return list(tokenize.tokenize(io.BytesIO(s.encode('utf-8')).readline))


def parse(tokens: List[Token]) -> SExpr:
    level = 0
    expressions: List[List[SExpr]] = []

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
    return parse(tokens)


def eval_expr(expr: SExpr, scope: Optional[Dict[str, SExpr]] = None):
    # TODO type checking
    if scope is None:
        scope: Dict[str, Value] = {}
    match expr:
        case () | Builtin.NIL:
            return ListValue(())
        case int() | float() | str() | bool():
            return expr
        case Builtin():
            return expr
        case Symbol(value=v):
            return scope[v]
        case (Builtin.IF, condition, if_true, if_false):
            condition_result = eval_expr(condition, scope)
            assert isinstance(condition_result, bool), f"if condition {print_expr(expr)} must be bool"
            if condition_result:
                return eval_expr(if_true, scope)
            else:
                return eval_expr(if_false, scope)
        case (Builtin.LET, Symbol(value=v), sub_expr):
            new_value = eval_expr(sub_expr, scope)
            scope[v] = new_value
            return new_value
        case (Builtin.LAMBDA, tuple(params), body):
            return Lambda(params=params, body=body, scope=scope)
        case (Builtin.SEXPR, *body):
            if len(body) != 1:
                raise ValueError(f"`sexpr` accepts only one argument, got multuple: {print_expr(expr)}")
            return ListValue(items=body)
        case (head, *args):
            head_eval = eval_expr(head, scope)
            match head_eval:
                case Builtin():
                    return apply_operator(head_eval, *[eval_expr(e, scope) for e in args])
                case Lambda():
                    lambda_scope = dict(head_eval.scope)
                    # TODO lazyiness
                    bound_params = {param.value: eval_expr(param_expr, scope) for param, param_expr in
                                    zip(head_eval.params, args)}
                    lambda_scope.update(**bound_params)
                    return eval_expr(head_eval.body, scope=lambda_scope)
                case _:
                    raise ValueError(f"Head expression evaluated to non-callable: {head}")
        case _:
            raise ValueError(f"Failed to evaluate expression: {print_expr(expr)}")


def apply_operator(op: Builtin, *args: Tuple[Value, ...]) -> Value:
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
            print("ouch!")
            return left < right
        case (Builtin.LTE, left, right):
            return left <= right
        case (Builtin.GT, left, right):
            return left > right
        case (Builtin.GTE, left, right):
            return left >= right
        case (Builtin.PRINT, arg):
            return print_expr(arg)
        case (Builtin.STRCAT, *args):
            return "".join(args)
        case (Builtin.ASSERT, condition):
            assert condition
            return True
        case (Builtin.ASSERT, condition, message):
            assert condition, message
            return True
        # List ops
        case (Builtin.LIST, *args):
            return ListValue(tuple(args))
        case (Builtin.CONS, head, ListValue(items=tail)):
            return ListValue((head, *tail))
        case (Builtin.HEAD | Builtin.TAIL, ListValue(items=items)) if len(items) == 0:
            raise ValueError(f"Can't call {op} on empty list")
        case (Builtin.HEAD, ListValue(items=(head, *_))):
            return head
        case (Builtin.TAIL, ListValue(items=(_, *tail))):
            return ListValue(tuple(tail))
        case _:
            raise ValueError(f"Failed to evaluate operator {print_expr((op, *args))}")


def print_expr(expr: SExpr) -> str:
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
        case tuple():
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
        (let add (lambda (x y) (+ x y)))
        (let mul (lambda (x y) (* x y)))
        (let sub (lambda (x y) (- x y)))
        (let neg (lambda (x) (- 0 x)))
        (let div (lambda (x y) (/ x y)))
        (let eq (lambda (x y) (= x y)))
        (let printf (lambda (x) (print x))) 
    """,
    "bools": """
        (let and (lambda (a b) (if a b a)))
        (let or (lambda (a b) (if a a b)))
        (let not (lambda (a) (if a false true)))
    """,
    "pairs": """
        (let pair (lambda (x y) (lambda (z) (z x y))))
        (let first (lambda (l) (l (lambda (x y) x)))) 
        (let second (lambda (l) (l (lambda (x y) y))))
    """,
    "list_ops": """
        (let is_nil (lambda (l) (= l nil)))
        (let fold (lambda (f l i) (if (is_nil l) i   (f (head l) (fold f (tail l) i)))))
        (let map (lambda (f l)   (if (is_nil l) nil (cons (f (head l)) (map f (tail l))))))
        (let map_val (lambda (i l) (map (lambda (_) i) l)))
        (let len (lambda (l) (fold add (map_val 1 l) 0)))
        (let cat (lambda (l1 l2) (if (is_nil l1) l2 (cons (head l1) (cat (tail l1) l2)))))
        (let range (lambda (start end) (and (assert (<= start end) (print (list start "<=" end))) (if (= start end) nil (cons start (range (+ start 1) end))))))
        (let flatten (lambda (l) (fold cat l nil)))
        (let cartesian_prod (lambda (l1 l2) (flatten (map (lambda (a) (map (lambda (b) (list a b)) l2)) l1))))
    """,

}


def load_preamble(preamble: str, scope: Dict[str, Literal]):
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

    scope = {}

    # load preambles
    # load_preamble(PREAMBLES["bools"], scope)
    for name in ["bools", "pairs", "list_ops", "builtins"]:
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
