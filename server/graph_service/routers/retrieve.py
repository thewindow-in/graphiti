from datetime import datetime, timezone
import logging
from typing import List, Optional, Set, Dict # Added Dict, Set

from fastapi import APIRouter, status, HTTPException
# Removed BaseModel, Field as they were only used for the removed endpoint models
# from pydantic import BaseModel, Field

from graph_service.dto import (
    GetMemoryRequest,
    GetMemoryResponse,
    Message,
    SearchQuery,
    SearchResults,
    FactResult,
)
from fastapi.params import Depends
from graph_service.auth import get_api_key
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import GroupsEdgesNotFoundError, NodeNotFoundError
# Import EpisodicNode to fetch episode details
from graphiti_core.nodes import EpisodicNode
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT
# Removed imports only needed for /find-groups
# from graphiti_core.search.search_config import SearchConfig
# from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graph_service.zep_graphiti import ZepGraphitiDep, get_fact_result_from_edge

router = APIRouter()

logger = logging.getLogger(__name__)

# Removed FindGroupsRequest and FindGroupsResponse models


@router.post('/search', status_code=status.HTTP_200_OK, response_model=SearchResults, dependencies=[Depends(get_api_key)])
async def search(query: SearchQuery, graphiti: ZepGraphitiDep):
    """
    Performs a hybrid search across the graph, optionally filtered by group_ids.
    """
    try:
        # Fix: graphiti.search returns only the list of edges
        relevant_edges: List[EntityEdge] = await graphiti.search(
            group_ids=query.group_ids,
            query=query.query,
            num_results=query.max_facts or DEFAULT_SEARCH_LIMIT,
        )

        # Fetch episode details needed for get_fact_result_from_edge
        episode_uuids_to_fetch: Set[str] = set()
        for edge in relevant_edges:
            if edge.episodes:
                episode_uuids_to_fetch.update(edge.episodes)

        episode_map: Dict[str, EpisodicNode] = {}
        if episode_uuids_to_fetch:
            try:
                # Fetch the actual EpisodicNode objects
                episode_nodes = await EpisodicNode.get_by_uuids(graphiti.driver, list(episode_uuids_to_fetch))
                # Create map from UUID to Node object
                episode_map = {node.uuid: node for node in episode_nodes}
                logger.debug(f"Fetched details for {len(episode_map)} episodes related to search results.")
            except NodeNotFoundError:
                logger.warning("Some episodes linked to search results were not found.")
            except Exception as e_fetch:
                 logger.error(f"Error fetching episode details for search results: {e_fetch}", exc_info=True)
                 # Proceed without episode details if fetching fails

        # Now create the FactResult list using the fetched episode details
        # Ensure get_fact_result_from_edge can handle potentially missing episodes in the map
        facts = [get_fact_result_from_edge(edge, episode_map) for edge in relevant_edges]

        logger.info(f"Search for '{query.query}' in groups {query.group_ids} returned {len(facts)} facts.")
        return SearchResults(facts=facts)
    except Exception as e:
        logger.error(f"Error during /search for query '{query.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to perform search.")


@router.get('/entity-edge/{uuid}', status_code=status.HTTP_200_OK, response_model=FactResult, dependencies=[Depends(get_api_key)])
async def get_entity_edge(uuid: str, graphiti: ZepGraphitiDep):
    """
    Retrieves a specific entity edge (fact) by its UUID.
    """
    try:
        entity_edge = await graphiti.get_entity_edge(uuid)
        # Fetch episode details for this specific edge
        episode_map: Dict[str, EpisodicNode] = {}
        if entity_edge.episodes:
            try:
                episode_nodes = await EpisodicNode.get_by_uuids(graphiti.driver, entity_edge.episodes)
                episode_map = {node.uuid: node for node in episode_nodes}
            except NodeNotFoundError:
                 logger.warning(f"Episodes {entity_edge.episodes} for edge {uuid} not found.")
            except Exception as e_fetch:
                 logger.error(f"Error fetching episode details for edge {uuid}: {e_fetch}", exc_info=True)
        # Pass the potentially populated map to the formatting function
        return get_fact_result_from_edge(entity_edge, episode_map)
    except HTTPException:
        raise # Re-raise 404 from get_entity_edge if not found
    except Exception as e:
        logger.error(f"Error retrieving entity edge {uuid}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve entity edge.")


@router.get('/episodes/{group_id}', status_code=status.HTTP_200_OK, response_model=List[Message], dependencies=[Depends(get_api_key)])
async def get_episodes(group_id: str, graphiti: ZepGraphitiDep, last_n: int = 10):
    """
    Retrieves the most recent 'last_n' episodes for a specific group_id.
    """
    if last_n <= 0:
        raise HTTPException(status_code=400, detail="'last_n' must be positive.")
    try:
        episodic_nodes = await graphiti.retrieve_episodes(
            group_ids=[group_id], last_n=last_n, reference_time=datetime.now(timezone.utc)
        )
        # Convert EpisodicNode to the Message DTO format
        messages = [
             Message(
                 uuid=ep_node.uuid,
                 role_type="episodic", # Indicate source type
                 content=ep_node.content,
                 timestamp=ep_node.valid_at,
                 name=ep_node.name,
                 source_description=ep_node.source_description,
                 # Map other fields from EpisodicNode to Message if your DTO requires them
             ) for ep_node in episodic_nodes
        ]
        return messages
    except Exception as e:
        logger.error(f"Error retrieving episodes for group {group_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve episodes.")


@router.post('/get-memory', status_code=status.HTTP_200_OK, response_model=GetMemoryResponse, dependencies=[Depends(get_api_key)])
async def get_memory(
    request: GetMemoryRequest,
    graphiti: ZepGraphitiDep,
):
    """
    Performs a search based on a list of messages, filtered by group_id.
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    try:
        combined_query = compose_query_from_messages(request.messages)
        logger.info(f"Composed query for get-memory in group {request.group_id}: '{combined_query[:200]}...'")

        # Fix: graphiti.search returns only the list of edges
        relevant_edges: List[EntityEdge] = await graphiti.search(
            group_ids=[request.group_id], # Pass group_id as a list
            query=combined_query,
            num_results=request.max_facts or DEFAULT_SEARCH_LIMIT,
        )

        # Fetch episode details needed for get_fact_result_from_edge
        episode_uuids_to_fetch: Set[str] = set()
        for edge in relevant_edges:
            if edge.episodes:
                episode_uuids_to_fetch.update(edge.episodes)

        episode_map: Dict[str, EpisodicNode] = {}
        if episode_uuids_to_fetch:
            try:
                episode_nodes = await EpisodicNode.get_by_uuids(graphiti.driver, list(episode_uuids_to_fetch))
                episode_map = {node.uuid: node for node in episode_nodes}
                logger.debug(f"Fetched details for {len(episode_map)} episodes related to get-memory results.")
            except NodeNotFoundError:
                 logger.warning("Some episodes linked to get-memory results were not found.")
            except Exception as e_fetch:
                 logger.error(f"Error fetching episode details for get-memory results: {e_fetch}", exc_info=True)

        # Create FactResult list
        facts = [get_fact_result_from_edge(edge, episode_map) for edge in relevant_edges]

        logger.info(f"Get-memory for group {request.group_id} returned {len(facts)} facts.")
        return GetMemoryResponse(facts=facts)
    except Exception as e:
        logger.error(f"Error during /get-memory for group {request.group_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get memory.")


@router.get('/group/{group_id}/facts', status_code=status.HTTP_200_OK, response_model=SearchResults, dependencies=[Depends(get_api_key)])
async def get_group_facts(
    group_id: str,
    graphiti: ZepGraphitiDep,
    limit: int = 20
):
    """
    Retrieves the most recent facts (entity edges) associated with a specific group_id.
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="'limit' must be positive.")
    try:
        logger.info(f"Fetching latest {limit} facts for group_id: {group_id}")
        edges = await EntityEdge.get_by_group_ids(
            driver=graphiti.driver,
            group_ids=[group_id],
            limit=limit
        )

        # Fetch episode details needed for get_fact_result_from_edge
        episode_uuids_to_fetch: Set[str] = set()
        for edge in edges:
            if edge.episodes:
                episode_uuids_to_fetch.update(edge.episodes)

        episode_map: Dict[str, EpisodicNode] = {}
        if episode_uuids_to_fetch:
            try:
                episode_nodes = await EpisodicNode.get_by_uuids(graphiti.driver, list(episode_uuids_to_fetch))
                episode_map = {node.uuid: node for node in episode_nodes}
                logger.debug(f"Fetched details for {len(episode_map)} episodes related to group facts.")
            except NodeNotFoundError:
                 logger.warning(f"Some episodes linked to group facts for {group_id} were not found.")
            except Exception as e_fetch:
                 logger.error(f"Error fetching episode details for group facts {group_id}: {e_fetch}", exc_info=True)

        # Create FactResult list
        facts = [get_fact_result_from_edge(edge, episode_map) for edge in edges]

        logger.info(f"Found {len(facts)} facts for group {group_id}.")
        return SearchResults(facts=facts)

    except GroupsEdgesNotFoundError:
        logger.info(f"No facts found for group_id: {group_id}")
        return SearchResults(facts=[]) # Return empty list if no edges found for group
    except Exception as e:
        logger.error(f"Error retrieving facts for group {group_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve facts for group {group_id}.")


def compose_query_from_messages(messages: list[Message]):
    """Helper function to combine messages into a single query string."""
    combined_query = ''
    for message in messages:
        # Use role_type if role is missing, default to 'user' if both missing
        role_prefix = message.role_type or message.role or "user"
        combined_query += f"{role_prefix}: {message.content}\n"
    return combined_query.strip()
