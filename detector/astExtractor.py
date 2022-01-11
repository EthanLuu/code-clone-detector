import ast


class CodeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.path = []

    def generic_visit(self, node):
        self.path.append(type(node).__name__)
        ast.NodeVisitor.generic_visit(self, node)

    def visit_FunctionDef(self, node):  # get funciton name
        self.path.append(type(node).__name__)

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Assign(self, node):  # get assign name
        self.path.append(type(node).__name__)

        ast.NodeVisitor.generic_visit(self, node)

    def visit_Name(self, node):  # get node.id
        self.path.append(node.id)


class AstExtractor:
    def __init__(self) -> None:
        self.visitor = CodeVisitor()

    def get_ast_string(self, code):
        tree = ast.parse(code)
        self.visitor.visit(tree)
        return self.visitor.path


def main():
    code = 'print(1)'
    astExtractor = AstExtractor()
    print(astExtractor.get_ast_string(code))


if __name__ == "__main__":
    main()
