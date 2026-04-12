import strawberry
from typing import List
from src.graph.knowledge_graph import KnowledgeGraph


@strawberry.type
class Entity:
    name: str
    kind: str


@strawberry.type
class Connection:
    target: str
    relation: str


@strawberry.type
class Query:

    @strawberry.field
    def entities(self) -> List[Entity]:
        kg = KnowledgeGraph()
        results = kg.get_all_entities()
        kg.close()
        return [Entity(name=r["name"], kind=r["kind"]) for r in results]

    @strawberry.field
    def entities_for_topic(self, topic: str) -> List[Entity]:
        kg = KnowledgeGraph()
        results = kg.get_entities_for_topic(topic)
        kg.close()
        return [Entity(name=r["name"], kind=r["kind"]) for r in results]

    @strawberry.field
    def connections(self, entity: str) -> List[Connection]:
        kg = KnowledgeGraph()
        results = kg.get_connections(entity)
        kg.close()
        return [Connection(target=r["target"], relation=r["relation"]) for r in results]


schema = strawberry.Schema(query=Query)
