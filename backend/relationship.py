from node import Node

class Relationship:
    def __init__(self, node1: Node, node2: Node, rel_label: str, rel_props: dict):
        self.node1 = node1
        self.node2 = node2
        self.rel_label = rel_label
        self.rel_props = rel_props
