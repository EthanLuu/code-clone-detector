import ast


class CodeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.path = []
        super().__init__()
        
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


def main():
    vistor = CodeVisitor()
    code = 'print(1)'
    tu = ast.parse(code)
    # print(ast.dump(tu))
    vistor.generic_visit(tu)


if __name__ == "__main__":
    main()
