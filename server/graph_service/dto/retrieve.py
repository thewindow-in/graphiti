from datetime import datetime, timezone
from typing import List

from pydantic import BaseModel, Field

from graph_service.dto.common import Message
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT


class SearchQuery(BaseModel):
    group_ids: list[str] | None = Field(
        None, description='The group ids for the memories to search'
    )
    query: str
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')


class FactResult(BaseModel):
    uuid: str
    name: str
    fact: str
    episodes: list[EpisodicNode]
    valid_at: datetime | None
    invalid_at: datetime | None
    created_at: datetime
    expired_at: datetime | None

    class Config:
        json_encoders = {datetime: lambda v: v.astimezone(timezone.utc).isoformat()}


class SearchResults(BaseModel):
    facts: list[FactResult]

# <<< CHANGE START >>>
# Define Request and Response models for the new endpoint
class FindGroupsRequest(BaseModel):
    query: str = Field(..., description="The user query or topic to find relevant groups for.")
    limit_results: int = Field(DEFAULT_SEARCH_LIMIT, description="Number of graph results to consider for extracting group IDs.")

class FindGroupsResponse(BaseModel):
    group_ids: List[str] = Field(..., description="A list of group IDs relevant to the query.")
# <<< CHANGE END >>>


class GetMemoryRequest(BaseModel):
    group_id: str = Field(..., description='The group id of the memory to get')
    max_facts: int = Field(default=10, description='The maximum number of facts to retrieve')
    center_node_uuid: str | None = Field(
        ..., description='The uuid of the node to center the retrieval on'
    )
    messages: list[Message] = Field(
        ..., description='The messages to build the retrieval query from '
    )


class GetMemoryResponse(BaseModel):
    facts: list[FactResult] = Field(..., description='The facts that were retrieved from the graph')
