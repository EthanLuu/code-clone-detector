from pycallgraph3 import PyCallGraph
from pycallgraph3.output import GraphvizOutput


class Banana:

    def eat(self):
        pass


class Person:

    def __init__(self):
        self.no_bananas()

    def no_bananas(self):
        self.bananas = []

    def add_banana(self, banana):
        self.bananas.append(banana)

    def eat_bananas(self):
        [banana.eat() for banana in self.bananas]
        self.no_bananas()


def main():
    graphviz = GraphvizOutput()
    graphviz.output_file = "output/call/banana.png"

    with PyCallGraph(output=graphviz):
        person = Person()
        for a in range(10):
            person.add_banana(Banana())
        person.eat_bananas()


if __name__ == '__main__':
    main()
