from typing import Tuple, Any, Iterable

import attr


class LinkedList:

    def __eq__(self, other):
        match (self, other):
            case Empty(), Empty():
                return True
            case Link(v1, r1), Link(v2, r2):
                return v1 == v2 and r1 == r2
            case _:
                return False

    def __len__(self):
        # NOTE! Can be expensive, traverses whole list
        return sum(1 for _ in self)

    @classmethod
    def of(cls, *args):
        match args:
            case ():
                return Empty()
            case (value, *rest):
                return Link(value, LinkedList.of(*rest))

    @classmethod
    def from_iterable(cls, iterable: Iterable) -> "LinkedList":
        dummy = Link(None, Empty())
        current = dummy
        for x in iterable:
            current.rest = Link(x, Empty())
            current = current.rest
        return dummy.rest

    @classmethod
    def linkify(cls, sexpr: Tuple | Any) -> Any | "LinkedList":
        match sexpr:
            case []:
                return cls.empty()
            case (_, *_):
                return cls.of(*map(cls.linkify, sexpr))
            case _:
                return sexpr

    @classmethod
    def tuplify(cls, sexpr: Any | "LinkedList"):
        match sexpr:
            case cls():
                return tuple(map(cls.tuplify, sexpr))
            case _:
                return sexpr


    @classmethod
    def empty(cls):
        return Empty()

    def __iter__(self):
        match self:
            case Empty():
                return
            case Link(value, rest):
                yield value
                yield from rest


class Empty(LinkedList):
    def __str__(self):
        return "()"


@attr.s(auto_detect=True)
class Link(LinkedList):
    __match_args__ = ("value", "rest")
    value = attr.ib()
    rest: LinkedList = attr.ib()

    def __str__(self):
        return f"({self.value} => {self.rest})"
