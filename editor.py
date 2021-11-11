import pprint
import traceback
from typing import List

from prompt_toolkit import PromptSession
from prompt_toolkit.layout.processors import HighlightMatchingBracketProcessor
from pygments import highlight
from pygments.formatters.terminal256 import Terminal256Formatter
from pygments.lexers.lisp import SchemeLexer
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.lexers import PygmentsLexer

from lisp import Interpreter


class Editor:

    def __init__(self, interpreter: Interpreter, prelude_filenames: List[str] = None):
        self.interpreter: Interpreter = interpreter
        self.prompt_session: PromptSession = None
        self.prelude_filenames = prelude_filenames or []
        self.init()

    def init(self):
        """
        Wipes program state & resets prompt session
        """
        self.interpreter.init()
        self.prompt_session = PromptSession(
            lexer=PygmentsLexer(SchemeLexer),
            input_processors=[HighlightMatchingBracketProcessor()]
        )

    def prompt(self) -> str:
        return self.prompt_session.prompt("> ", lexer=PygmentsLexer(SchemeLexer),
                                          input_processors=[HighlightMatchingBracketProcessor()])

    def print(self, s: str):
        print(highlight(s, SchemeLexer(), Terminal256Formatter()).strip())

    def run(self):
        self.init()

        for fn in self.prelude_filenames:
                self.load_file(fn)

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
