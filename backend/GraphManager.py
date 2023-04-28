from neo4j import GraphDatabase
from node import Node
import os
from relationship import Relationship

class GraphManager:

    def __init__(self, uri: str | None = None, username: str | None = None, password: str | None = None):
        if uri == None:
            uri = os.getenv("NEO4J_URI")
        if uri == None:
            raise "No URI provided."
        if username == None:
            username = os.getenv("NEO4J_USERNAME")
        if username == None:
            raise "No username provided."
        if password == None:
            password = os.getenv("NEO4J_PASSWORD")
        if password == None:
            raise "No password provided."
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    
    def stringify_props(self, props: dict):
        if props == {}:
            return "{}"
        string_props = "{"
        for k,v in props.items():
            if isinstance(v, str) or v == None:
                string_props += k + ": '" + str(v) + "'" + ", "
            elif isinstance(v, int) or isinstance(v, float):
                string_props += k + ": " + v + ", "
            elif isinstance(v, map):
                continue
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

    def create_node(self, tx, node: Node):
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

    def wipe(self, tx):
        tx.run(
            "MATCH (n) "
            "DETACH DELETE n"
        )
        print("successfully wiped database")

    def close(self):
        self.driver.close()