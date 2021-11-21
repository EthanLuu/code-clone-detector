import ast


class AstExtractor:
    def get_ast_string(self, code):
        tree = ast.parse(code)
        return ast.dump(tree)


def main():
    code = 'print(1)'
    astExtractor = AstExtractor()
    print(astExtractor.get_ast_string(code))


if __name__ == "__main__":
    main()
