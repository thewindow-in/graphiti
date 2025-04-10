"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

MODIFIED FOR EFFICIENT BATCH COMMUNITY SUMMARIZATION AND ROBUSTNESS
"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional # Added List, Tuple, Optional

from neo4j import AsyncDriver
from pydantic import BaseModel

from graphiti_core.edges import CommunityEdge
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import DEFAULT_DATABASE, semaphore_gather
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    get_community_node_from_record,
    get_entity_node_from_record, # Import needed if get_community_clusters is modified
)
from graphiti_core.prompts import prompt_library
# <<< CHANGE START >>>
# Import the new response model for batch summarization
from graphiti_core.prompts.summarize_nodes import CommunitySummaryAndName, Summary, SummaryDescription
# <<< CHANGE END >>>
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import build_community_edges

MAX_COMMUNITY_BUILD_CONCURRENCY = 10 # Adjust as needed based on LLM rate limits and server capacity

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


async def get_community_clusters(
    driver: AsyncDriver, group_ids: list[str] | None
) -> list[list[EntityNode]]:
    """
    Finds community clusters within specified group IDs using label propagation.
    Consider replacing with Neo4j GDS library for large graphs if performance is an issue.
    e.g., CALL gds.labelPropagation.stream('myGraph') YIELD nodeId, communityId
    """
    logger.info(f"Starting community detection for group_ids: {group_ids or 'all'}")
    community_clusters_by_uuid: Dict[int, List[str]] = defaultdict(list)
    all_nodes_map: Dict[str, EntityNode] = {} # Store nodes to avoid re-fetching

    if group_ids is None:
        logger.debug("No group IDs specified, fetching all distinct group IDs.")
        group_id_records, _, _ = await driver.execute_query(
            """
            MATCH (n:Entity) WHERE n.group_id IS NOT NULL
            RETURN collect(DISTINCT n.group_id) AS group_ids
            """,
            database_=DEFAULT_DATABASE,
            routing_='r', # Use read routing
        )
        group_ids = group_id_records[0]['group_ids'] if group_id_records and group_id_records[0]['group_ids'] else []
        logger.debug(f"Found group IDs: {group_ids}")

    if not group_ids:
        logger.warning("No group IDs found or specified for community detection.")
        return []

    # Fetch all nodes for the relevant groups once
    logger.debug(f"Fetching all entity nodes for groups: {group_ids}")
    all_nodes = await EntityNode.get_by_group_ids(driver, group_ids)
    for node in all_nodes:
        all_nodes_map[node.uuid] = node
    logger.debug(f"Fetched {len(all_nodes_map)} nodes.")


    # --- Client-Side Label Propagation ---
    # Note: This can be memory intensive and slow for very large graphs.
    # Consider Neo4j GDS library (gds.labelPropagation.stream) for better performance.

    # Build the projection (adjacency list with weights)
    projection: Dict[str, List[Neighbor]] = {uuid: [] for uuid in all_nodes_map}
    logger.debug("Building graph projection for label propagation...")
    # This query might be slow if run per node. Consider a single query if possible.
    # Example single query (adjust relationship type and direction if needed):
    # MATCH (n:Entity)-[r:RELATES_TO]-(m:Entity)
    # WHERE n.group_id IN $group_ids AND m.group_id IN $group_ids
    # RETURN n.uuid AS source, m.uuid AS target, count(r) AS weight
    # This requires processing the results into the projection dict.

    # Using per-node query for simplicity based on original code:
    neighbor_results = await semaphore_gather(*[
         driver.execute_query(
                """
                MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-(m: Entity)
                WHERE n.group_id = m.group_id AND n.group_id IN $group_ids
                RETURN m.uuid AS neighbor_uuid, count(r) AS edge_count
                """,
                uuid=node_uuid,
                group_ids=group_ids, # Pass group_ids for filtering
                database_=DEFAULT_DATABASE,
                routing_='r',
            )
         for node_uuid in all_nodes_map.keys()
    ])

    for i, node_uuid in enumerate(all_nodes_map.keys()):
        records, _, _ = neighbor_results[i]
        projection[node_uuid] = [
            Neighbor(node_uuid=record['neighbor_uuid'], edge_count=record['edge_count'])
            for record in records
        ]
    logger.debug("Graph projection built.")


    # Run Label Propagation
    logger.debug("Running label propagation...")
    clusters_by_uuid = label_propagation(projection) # Returns dict {community_id: [node_uuid1, ...]}
    logger.debug(f"Label propagation finished. Found {len(clusters_by_uuid)} potential communities.")


    # Convert UUID clusters back to EntityNode clusters
    community_clusters: list[list[EntityNode]] = []
    for cluster_uuids in clusters_by_uuid.values():
        # Filter out clusters with only one node (optional, adjust as needed)
        if len(cluster_uuids) > 1:
            cluster_nodes = [all_nodes_map[uuid] for uuid in cluster_uuids if uuid in all_nodes_map]
            if cluster_nodes: # Ensure the cluster is not empty after filtering
                 community_clusters.append(cluster_nodes)

    logger.info(f"Community detection complete. Identified {len(community_clusters)} clusters with >1 node.")
    return community_clusters


def label_propagation(projection: Dict[str, List[Neighbor]]) -> Dict[int, List[str]]:
    """Performs label propagation. Returns a map of community_id to list of node UUIDs."""
    if not projection:
        return {}

    # Initialize communities: each node is its own community initially
    community_map: Dict[str, int] = {uuid: i for i, uuid in enumerate(projection.keys())}
    iteration = 0
    max_iterations = 10 # Add a max iteration limit to prevent infinite loops

    while iteration < max_iterations:
        iteration += 1
        logger.debug(f"Label propagation iteration {iteration}")
        no_change = True
        new_community_map: Dict[str, int] = {}

        # Process nodes in a consistent order (e.g., sorted UUIDs) for determinism
        sorted_uuids = sorted(projection.keys())

        for uuid in sorted_uuids:
            neighbors = projection.get(uuid, [])
            if not neighbors:
                new_community_map[uuid] = community_map[uuid] # Node keeps its own community if isolated
                continue

            # Tally neighbor communities and weights
            community_candidates: Dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                neighbor_uuid = neighbor.node_uuid
                if neighbor_uuid in community_map: # Ensure neighbor is in the map
                     neighbor_community = community_map[neighbor_uuid]
                     community_candidates[neighbor_community] += neighbor.edge_count # Use edge count as weight

            if not community_candidates:
                 # If no neighbors had communities (shouldn't happen if projection is complete)
                 new_community_map[uuid] = community_map[uuid]
                 continue

            # Find the community with the highest total weight
            # Sort by weight (desc), then community ID (asc) for tie-breaking
            sorted_candidates = sorted(community_candidates.items(), key=lambda item: (-item[1], item[0]))
            best_community = sorted_candidates[0][0]

            # Update node's community
            new_community_map[uuid] = best_community
            if best_community != community_map[uuid]:
                no_change = False

        community_map = new_community_map # Update map for next iteration

        if no_change:
            logger.debug(f"Label propagation converged after {iteration} iterations.")
            break
    else:
         logger.warning(f"Label propagation did not converge after {max_iterations} iterations.")


    # Group nodes by final community ID
    final_clusters: Dict[int, List[str]] = defaultdict(list)
    for uuid, community_id in community_map.items():
        final_clusters[community_id].append(uuid)

    return final_clusters


async def build_community(
    llm_client: LLMClient, community_cluster: list[EntityNode]
) -> Optional[Tuple[CommunityNode, list[CommunityEdge]]]: # Return Optional in case of failure
    """Builds a single CommunityNode and its edges from a cluster of EntityNodes."""
    if not community_cluster:
        return None

    start_time = utc_now()
    first_node = community_cluster[0] # Use for group_id and fallback naming
    logger.debug(f"Building community for cluster starting with node: {first_node.uuid} ({first_node.name}), size: {len(community_cluster)}")

    # Prepare context for the batch summarization LLM call
    # Include names and potentially truncated summaries
    # Limit the number of members sent to LLM to avoid overly large prompts
    MAX_MEMBERS_FOR_LLM = 50
    members_context = [
        {"name": node.name, "summary": node.summary or ""}
        for node in community_cluster[:MAX_MEMBERS_FOR_LLM]
    ]
    context = {'community_members': members_context}

    community_name = f"Community around {first_node.name}" # Fallback name
    community_summary = f"A community cluster containing {len(community_cluster)} nodes, including {first_node.name}." # Fallback summary

    try:
        # <<< CHANGE START >>>
        # Use the new batch summarization prompt
        llm_response = await llm_client.generate_response(
            prompt_library.summarize_nodes.summarize_community_batch(context),
            response_model=CommunitySummaryAndName
        )
        # Ensure response is a dict before accessing
        response_data = llm_response if isinstance(llm_response, dict) else {}
        community_name = response_data.get('community_name', community_name) # Use LLM name or fallback
        community_summary = response_data.get('community_summary', community_summary) # Use LLM summary or fallback
        logger.debug(f"LLM generated community name: '{community_name}', summary: '{community_summary}'")
        # <<< CHANGE END >>>
    except Exception as e:
        logger.error(f"Failed to generate community summary/name via LLM for cluster starting with {first_node.uuid}: {e}", exc_info=True)
        logger.warning("Using fallback naming/summary for community.")
        # Fallback values are already set above

    # Create the Community Node
    community_node = CommunityNode(
        name=community_name,
        group_id=first_node.group_id, # Assume all nodes in cluster share group_id
        labels=['Community'], # Add Community label
        created_at=start_time,
        summary=community_summary,
        # name_embedding will be generated later
    )

    # Create edges linking entities to the new community node
    community_edges = build_community_edges(community_cluster, community_node, start_time)

    logger.debug(f"Built CommunityNode {community_node.uuid} ('{community_node.name}') with {len(community_edges)} member edges.")

    return community_node, community_edges


async def build_communities(
    driver: AsyncDriver, llm_client: LLMClient, group_ids: list[str] | None
) -> tuple[list[CommunityNode], list[CommunityEdge]]:
    """Detects clusters and builds CommunityNodes for them."""
    # 1. Detect clusters
    # Consider adding error handling around cluster detection itself
    try:
        community_clusters = await get_community_clusters(driver, group_ids)
    except Exception as e:
        logger.error(f"Error during community cluster detection: {e}", exc_info=True)
        return [], [] # Return empty if detection fails

    if not community_clusters:
         logger.info("No community clusters found to build.")
         return [], []

    # 2. Build CommunityNode for each cluster concurrently
    semaphore = asyncio.Semaphore(MAX_COMMUNITY_BUILD_CONCURRENCY)

    async def limited_build_community(cluster):
        async with semaphore:
            # Add error handling for individual community builds
            try:
                return await build_community(llm_client, cluster)
            except Exception as e:
                 logger.error(f"Error building individual community (cluster size {len(cluster)}): {e}", exc_info=True)
                 return None # Return None if build fails for this cluster

    # Run build tasks concurrently
    build_results: List[Optional[Tuple[CommunityNode, List[CommunityEdge]]]] = list(
        await semaphore_gather(
            *[limited_build_community(cluster) for cluster in community_clusters]
        )
    )

    # 3. Collect successful results
    community_nodes: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    successful_builds = 0
    for result in build_results:
        if result: # Check if result is not None (i.e., build succeeded)
            node, edges = result
            community_nodes.append(node)
            community_edges.extend(edges)
            successful_builds += 1

    logger.info(f"Successfully built {successful_builds}/{len(community_clusters)} communities.")
    return community_nodes, community_edges


async def remove_communities(driver: AsyncDriver):
    """Removes all Community nodes and their associated HAS_MEMBER edges."""
    logger.warning("Removing all existing Community nodes and HAS_MEMBER edges.")
    try:
        # Use DETACH DELETE to remove nodes and their relationships
        await driver.execute_query(
            """
            MATCH (c:Community)
            DETACH DELETE c
            """,
            database_=DEFAULT_DATABASE,
        )
        logger.info("Successfully removed existing communities.")
    except Exception as e:
        logger.error(f"Error removing communities: {e}", exc_info=True)
        # Decide if this should raise an error or just log


async def determine_entity_community(
    driver: AsyncDriver, entity: EntityNode
) -> tuple[CommunityNode | None, bool]:
    """Determines if an entity belongs to a community, or finds the best fit."""
    # Check if the node is already part of a community
    existing_community_records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
        RETURN c.uuid AS uuid, c.name AS name, c.name_embedding AS name_embedding,
               c.group_id AS group_id, c.created_at AS created_at, c.summary AS summary
        LIMIT 1
        """,
        entity_uuid=entity.uuid,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    if existing_community_records:
        logger.debug(f"Entity {entity.uuid} already belongs to community {existing_community_records[0]['uuid']}")
        return get_community_node_from_record(existing_community_records[0]), False

    # Find the mode community among neighbors
    neighbor_community_records, _, _ = await driver.execute_query(
        """
        MATCH (n:Entity {uuid: $entity_uuid})-[:RELATES_TO]-(neighbor:Entity)
        MATCH (c:Community)-[:HAS_MEMBER]->(neighbor)
        WHERE n.group_id = neighbor.group_id // Ensure neighbors are in the same group
        RETURN c.uuid AS uuid, c.name AS name, c.name_embedding AS name_embedding,
               c.group_id AS group_id, c.created_at AS created_at, c.summary AS summary,
               count(c) AS community_count
        ORDER BY community_count DESC
        LIMIT 1 // Get the most frequent neighbor community
        """,
        entity_uuid=entity.uuid,
        database_=DEFAULT_DATABASE,
        routing_='r',
    )

    if neighbor_community_records:
        best_fit_community = get_community_node_from_record(neighbor_community_records[0])
        logger.debug(f"Entity {entity.uuid} best fits into community {best_fit_community.uuid} based on neighbors.")
        return best_fit_community, True # True indicates it's a new assignment
    else:
        logger.debug(f"Could not determine a community for entity {entity.uuid} based on neighbors.")
        return None, False


async def update_community(
    driver: AsyncDriver, llm_client: LLMClient, embedder: EmbedderClient, entity: EntityNode
):
    """Adds an entity to its best-fit community and updates the community summary/name."""
    logger.debug(f"Attempting to update community based on entity: {entity.uuid} ({entity.name})")
    community, is_new_member = await determine_entity_community(driver, entity)

    if community is None:
        logger.debug(f"No suitable community found for entity {entity.uuid}. Skipping update.")
        return

    # Add edge if it's a new member assignment
    if is_new_member:
        logger.info(f"Assigning entity {entity.uuid} as new member to community {community.uuid}")
        community_edge = (build_community_edges([entity], community, utc_now()))[0]
        try:
             await community_edge.save(driver)
        except Exception as e:
             logger.error(f"Failed to save new membership edge for entity {entity.uuid} to community {community.uuid}: {e}", exc_info=True)
             # Continue with summary update even if edge fails? Or return? Decide policy.
             return # Let's return if we can't even add the membership edge

    # Update community summary and name by incorporating the new entity's info
    # Use the batch prompt for consistency, even with just two items? Or use summarize_pair?
    # Using summarize_pair might be slightly more efficient here.
    logger.debug(f"Updating summary for community {community.uuid} with info from entity {entity.uuid}")
    try:
        # Prepare context for summarizing the existing community and the new entity
        context = {'node_summaries': [
             {"summary": community.summary},
             {"summary": entity.summary}
        ]}
        # Use summarize_pair as it's designed for two inputs
        llm_summary_resp = await llm_client.generate_response(
            prompt_library.summarize_nodes.summarize_pair(context),
            response_model=Summary
        )
        new_summary = llm_summary_resp.get('summary', community.summary) # Fallback to old summary

        # Generate a new name based on the updated summary
        llm_name_resp = await llm_client.generate_response(
             prompt_library.summarize_nodes.summary_description({'summary': new_summary}),
             response_model=SummaryDescription
        )
        new_name = llm_name_resp.get('description', community.name) # Fallback to old name

        community.summary = new_summary
        community.name = new_name

        # Regenerate embedding for the updated community node
        await community.generate_name_embedding(embedder)

        # Save the updated community node
        await community.save(driver)
        logger.info(f"Successfully updated community {community.uuid} with info from entity {entity.uuid}")

    except Exception as e:
        logger.error(f"Failed to update community {community.uuid} summary/name/embedding via LLM: {e}", exc_info=True)
        # Community node might be saved without updated summary/name if LLM fails

