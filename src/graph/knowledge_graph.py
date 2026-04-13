import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")


class KnowledgeGraph:
    def __init__(self):
        # don't connect on init — connect lazily when first used
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(NEO4J_URI, auth=None)
        return self._driver

    def close(self):
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def clear(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def add_topic(self, topic: str):
        with self.driver.session() as session:
            session.run(
                "MERGE (t:Topic {name: $name})",
                name=topic
            )

    def add_entity(self, name: str, kind: str):
        with self.driver.session() as session:
            session.run(
                "MERGE (e:Entity {name: $name, kind: $kind})",
                name=name, kind=kind
            )

    def link_entity_to_topic(self, entity: str, topic: str, relation: str = "RELATED_TO"):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (e:Entity {name: $entity})
                MATCH (t:Topic {name: $topic})
                MERGE (e)-[:RELATED_TO {type: $relation}]->(t)
                """,
                entity=entity, topic=topic, relation=relation
            )

    def link_entities(self, source: str, target: str, relation: str):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[:CONNECTED {type: $relation}]->(b)
                """,
                source=source, target=target, relation=relation
            )

    def get_entities_for_topic(self, topic: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)-[:RELATED_TO]->(t:Topic {name: $topic})
                RETURN e.name AS name, e.kind AS kind
                """,
                topic=topic
            )
            return [{"name": r["name"], "kind": r["kind"]} for r in result]

    def get_all_entities(self) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) RETURN e.name AS name, e.kind AS kind"
            )
            return [{"name": r["name"], "kind": r["kind"]} for r in result]

    def get_connections(self, entity: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (a:Entity {name: $entity})-[r]->(b)
                RETURN b.name AS target, type(r) AS relation
                """,
                entity=entity
            )
            return [{"target": r["target"], "relation": r["relation"]} for r in result]
