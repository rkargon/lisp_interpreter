import pprint
import traceback

from lisp import Interpreter


class Editor:

    def __init__(self, interpreter: Interpreter):
        self.interpreter = interpreter

        import readline
        readline.read_init_file("./editrc")

    def prompt(self) -> str:
        # TODO use pygments
        return input("> ")

    def run(self):
        # wipe state
        self.interpreter.init()

        # load stdlib
        self.interpreter.load_file("./samples/stdlib.lisp")

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
                            self.interpreter.load_file(filename)
                            pass
                        case _:
                            raise ValueError(f"Could not understand command {string}")
                    continue
                else:
                    expression = self.interpreter.read(string)
                    if expression is None:
                        continue
                    result = self.interpreter.eval(expression)
                    print(self.interpreter.print(result))
            except Exception as e:
                traceback.print_exception(e)
