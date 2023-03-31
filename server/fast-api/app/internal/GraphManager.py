from neo4j import GraphDatabase
import numpy
from app.internal.node import Node
from app.internal.relationship import Relationship

class GraphManager:

    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    
    def stringify_props(self, props: dict):
        if props == {}:
            return "{}"
        string_props = "{"
        for k,v in props.items():
            if isinstance(v, str):
                string_props += k + ": '" + v + "'" + ", "
            elif isinstance(v, int) or isinstance(v, float):
                string_props += k + ": " + v + ", "
            elif isinstance(v, numpy.ndarray):
                new_arr = v.flatten()
                string_props += k + ": '" + str(new_arr)[1:-2].replace("'", "*") + "', "
            else:
                string_props += k + ": " + str(v) + ", "
        string_props = string_props[:-2]
        string_props += "}"
        return string_props

    def execute(self, **params):
        with self.driver.session() as session:
            if params.get("operation") == "create_node":
                node = params.get("node")
                session.execute_write(self.create_node, node)
            elif params.get("operation") == "create_rel":
                rel = params.get("rel")
                session.execute_write(self.create_rel, rel)
            elif params.get("operation") == "wipe":
                session.execute_write(self.wipe)
            elif params.get("operation") == "search":
                query = params.get("query")
                name = params.get("name")
                label = params.get("label")
                property = params.get("property")
                session.execute_write(self.search, query, name, label, property)

    def create_node(self, tx, node: Node):
        for k in node.properties.keys():
            if not isinstance(node.properties[k], numpy.ndarray):
                if node.properties[k] == None:
                    node.properties[k] = ""
        node_props_string = self.stringify_props(node.properties)
        result = tx.run(
            f"MATCH (p:{node.label} {node_props_string})"
            "RETURN COUNT (p) as cnt"
        )
        if result.single()["cnt"] == 0:
            tx.run(
                f"CREATE (a:{node.label} {node_props_string})"
            )
            print("successfully created node")
        else:
            print("node already exists")

    def create_rel(self, tx, rel: Relationship):
        node1 = rel.node1
        node2 = rel.node2
        node1_props_string = self.stringify_props(node1.properties)
        node2_props_string = self.stringify_props(node2.properties)
        rel_props_string = self.stringify_props(rel.rel_props)
        result = tx.run(
            f"MATCH (a:{node1.label} {node1_props_string})"
            f"MATCH (b:{node2.label} {node2_props_string})"
            f"MATCH (a)-[r:{rel.rel_label} {rel_props_string}]->(b)"
            "RETURN COUNT (r) as cnt"
        )
        if result.single()["cnt"] == 0:
            tx.run(
                f"MATCH (a:{node1.label} {node1_props_string})"
                f"MATCH (b:{node2.label} {node2_props_string})"
                f"CREATE (a)-[r:{rel.rel_label} {rel_props_string}]->(b)"
            )
            print("successfully created relationship")
        else:
            print("relationship already exists")
    
    def search(self, tx, query, name, label, property):
        print("running now")
        result = tx.run(
            f"CREATE FULLTEXT INDEX {name} IF NOT EXISTS FOR (n:{label}) ON EACH [n.{property}]"
            # f'CALL db.index.fulltext.queryNodes("{name}", "{query}") YIELD node, score '
            # f"RETURN node.{property}, score"
        )
        print(result.single())
        return result

    def wipe(self, tx):
        tx.run(
            "MATCH (n) "
            "DETACH DELETE n"
        )
        print("successfully wiped database")

    def close(self):
        self.driver.close()


# if __name__ == "__main__":
#     gp = GraphPopulator("neo4j+ssc://39f470cd.databases.neo4j.io", "neo4j","GRQHnkLdqja2PXzjYUg4wwoCxCI3uAGPOjbe6N_K6KM")
#     bp_node = Node('BP', {'BP_number': '0001'})
#     specReq_node = Node('SpecialRequirement', {'Requirements': ["req1", "req2", "req3"]})
#     rel1 = Relationship(bp_node, specReq_node, "HAS", {})
#     gp.execute(operation="create_rel", rel=rel1)
#     gp.close()
