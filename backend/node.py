class Node:
    def __init__(self, label: str, properties: dict = {}):
        self.label = label
        self.properties = properties

    def add_property(self, key, value):
        self.properties[key] = value


# sample populator
# bp_node = Node('BP', {'BP_number': '0001'})
# specReq_node = Node('SpecialRequirement', {'Requirements': ["req1", "req2", "req3"]})
# node_list = [bp_node, specReq_node]
