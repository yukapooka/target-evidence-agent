from pydantic import BaseModel

from agent.tools.http import graphql

SEARCH_DISEASE_QUERY = """
query SearchDisease($query: String!, $size: Int!) {
  search(queryString: $query, entityNames: ["disease"], page: {index: 0, size: $size}) {
    hits { id name description }
  }
}
"""

class DiseaseHit(BaseModel):
    id: str
    name: str
    description: str | None = None

def search_disease(query: str, size: int = 5) -> list[DiseaseHit]:
    """Search Open Targets for diseases matching a name; returns the top hits for the caller to choose from."""
    data = graphql(SEARCH_DISEASE_QUERY, {"query": query, "size": size})
    return [DiseaseHit(**hit) for hit in data["search"]["hits"]]