import pprint
import traceback

from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.lisp import SchemeLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer

from lisp import Interpreter


class Editor:

    def __init__(self, interpreter: Interpreter):
        self.interpreter = interpreter

        import readline
        readline.read_init_file("./editrc")

    def prompt(self) -> str:
        # TODO use pygments
        return prompt("> ", lexer=PygmentsLexer(SchemeLexer), input_processors=[HighlightMatchingBracketProcessor()])

    def print(self, s: str):
        print(highlight(s, SchemeLexer(), Terminal256Formatter()).strip())

    def run(self):
        # wipe state
        self.interpreter.init()

        # load stdlib
        self.load_file("./samples/stdlib.lisp")

        while True:
            string = self.prompt()
            if not string:
                continue
            try:
                # parse special commands
                if string.startswith("!"):
                    match string.split():
                        case "!load", *_:
                            filename = string.replace("!load ", "").strip()
                            print("Loading ", filename, "...")
                            self.load_file(filename)
                            pass
                        case _:
                            raise ValueError(f"Could not understand command {string}")
                    continue
                else:
                    expression = self.interpreter.read(string)
                    if expression is None:
                        continue
                    result = self.interpreter.eval(expression)
                    self.print(self.interpreter.print(result))
            except Exception as e:
                traceback.print_exception(e)

    def load_file(self, filename: str):
        with open(filename, "r") as f:
            for l in f.readlines():
                lstrip = l.strip()
                if not lstrip:
                    continue
                self.print("> " + lstrip)
                self.interpreter.repl(lstrip)
