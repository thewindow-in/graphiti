# graphiti_core/graphiti.py

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

MODIFIED TO HANDLE:
- Setting created_at from reference_time (original message time).
- Accepting and storing source channel and human/bot status.
- Passing episode context to helper functions for correct timestamping.
- Accepting priority parameters in search.
- Removed reliance on bulk operations.
"""

from http.client import HTTPException
import logging
from datetime import datetime
from time import time
from typing import List, Optional, Dict, Set # Added Dict, Set
from uuid import uuid4 # Added uuid4

from dotenv import load_dotenv
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError, GroupsEdgesNotFoundError # Added GroupsEdgesNotFoundError
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge, EpisodicEdge, CommunityEdge # Added CommunityEdge
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.helpers import DEFAULT_DATABASE, semaphore_gather
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search import SearchConfig, search # Renamed internal function to avoid conflict
from graphiti_core.search.search_config import DEFAULT_SEARCH_LIMIT, SearchResults
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_NODE_DISTANCE,
    EDGE_HYBRID_SEARCH_RRF,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    RELEVANT_SCHEMA_LIMIT,
    get_mentioned_nodes,
    get_relevant_edges,
    get_relevant_nodes,
)
# --- Removed bulk_utils imports ---
# from graphiti_core.utils.bulk_utils import (
#     RawEpisode,
#     add_nodes_and_edges_bulk,
#     dedupe_edges_bulk,
#     extract_edge_dates_bulk,
#     extract_nodes_and_edges_bulk,
#     resolve_edge_pointers,
#     retrieve_previous_episodes_bulk,
# )
# --- Add import for add_nodes_and_edges_bulk ---
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk, resolve_edge_pointers

from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    dedupe_extracted_edge, # Keep for add_triplet
    extract_edges,
    resolve_edge_contradictions, # Keep for add_triplet
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_node, # Keep for add_triplet
    resolve_extracted_nodes,
)
from graphiti_core.utils.maintenance.temporal_operations import get_edge_contradictions # Keep for add_triplet

logger = logging.getLogger(__name__)

load_dotenv()


class AddEpisodeResults(BaseModel):
    episode: EpisodicNode
    nodes: list[EntityNode]
    edges: list[EntityEdge]


class Graphiti:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        llm_client: LLMClient | None = None,
        embedder: EmbedderClient | None = None,
        cross_encoder: CrossEncoderClient | None = None,
        store_raw_episode_content: bool = True,
    ):
        """
        Initialize a Graphiti instance.

        This constructor sets up a connection to the Neo4j database and initializes
        the LLM client for natural language processing tasks.

        Parameters
        ----------
        uri : str
            The URI of the Neo4j database.
        user : str
            The username for authenticating with the Neo4j database.
        password : str
            The password for authenticating with the Neo4j database.
        llm_client : LLMClient | None, optional
            An instance of LLMClient for natural language processing tasks.
            If not provided, a default OpenAIClient will be initialized.
        embedder : EmbedderClient | None, optional
            An instance of EmbedderClient for generating embeddings.
            If not provided, a default OpenAIEmbedder will be initialized.
        cross_encoder : CrossEncoderClient | None, optional
            An instance of CrossEncoderClient for reranking search results.
            If not provided, a default OpenAIRerankerClient will be initialized.
        store_raw_episode_content : bool, optional
            Whether to store the full content of episodic nodes. Defaults to True.

        Returns
        -------
        None
        """
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = DEFAULT_DATABASE
        self.store_raw_episode_content = store_raw_episode_content
        if llm_client:
            self.llm_client = llm_client
        else:
            self.llm_client = OpenAIClient() # Consider making default configurable
        if embedder:
            self.embedder = embedder
        else:
            self.embedder = OpenAIEmbedder() # Consider making default configurable
        if cross_encoder:
            self.cross_encoder = cross_encoder
        else:
            self.cross_encoder = OpenAIRerankerClient() # Consider making default configurable

    async def close(self):
        """
        Close the connection to the Neo4j database.
        """
        await self.driver.close()

    async def build_indices_and_constraints(self, delete_existing: bool = False):
        """
        Build indices and constraints in the Neo4j database.
        """
        await build_indices_and_constraints(self.driver, delete_existing)

    async def retrieve_episodes(
        self,
        reference_time: datetime,
        last_n: int = EPISODE_WINDOW_LEN,
        group_ids: list[str] | None = None,
    ) -> list[EpisodicNode]:
        """
        Retrieve the last n episodic nodes from the graph.
        """
        effective_group_ids = group_ids if group_ids else None
        return await retrieve_episodes(self.driver, reference_time, last_n, effective_group_ids)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime, # This is the original Slack message timestamp (ts)
        source: EpisodeType = EpisodeType.message,
        group_id: str = '',
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, BaseModel] | None = None,
        # --- MODIFICATION START: Add channel/human status ---
        source_channel_id: Optional[str] = None,
        source_channel_name: Optional[str] = None,
        is_human_message: bool = True,
        # --- MODIFICATION END ---
    ) -> AddEpisodeResults:
        """
        Process an episode and update the graph.
        Handles node/edge extraction, deduplication, embedding, temporal resolution,
        and saving to the graph. Includes logic for handling recurring dynamic events.
        Uses original message time (`reference_time`) for `created_at` of new elements.
        """
        try:
            start_time_processing = time() # Track processing time
            all_edges_to_save: list[EntityEdge] = [] # Collect all entity edges here

            # Ensure group_id is treated consistently
            current_group_id = group_id or 'default'

            # Fetch previous episodes for context
            previous_episodes = await self.retrieve_episodes(
                reference_time, last_n=RELEVANT_SCHEMA_LIMIT, group_ids=[current_group_id]
            )

            # Create or retrieve the episodic node
            episode: Optional[EpisodicNode] = None
            if uuid:
                 try:
                     episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
                     # Update existing episode fields if needed
                     episode.content = episode_body
                     episode.valid_at = reference_time
                     episode.source_channel_id = source_channel_id
                     episode.source_channel_name = source_channel_name
                     episode.is_human_message = is_human_message
                     # Note: We don't update created_at for existing nodes
                     logger.debug(f"Found and updated existing episode {uuid}")
                 except NodeNotFoundError:
                      logger.warning(f"Episode UUID {uuid} provided but not found. Creating new episode.")
                      uuid = None # Fallback to creating a new one

            if not episode: # Create new if no UUID or if lookup failed
                episode_uuid = uuid if uuid else str(uuid4())
                episode = EpisodicNode(
                    uuid=episode_uuid,
                    name=name,
                    group_id=current_group_id,
                    labels=[],
                    source=source,
                    content=episode_body,
                    source_description=source_description,
                    # --- MODIFICATION START: Set created_at and new fields ---
                    created_at=reference_time, # Use original message time
                    valid_at=reference_time,
                    source_channel_id=source_channel_id,
                    source_channel_name=source_channel_name,
                    is_human_message=is_human_message,
                    # --- MODIFICATION END ---
                )
                logger.debug(f"Created new episode {episode.uuid}")


            # --- 1. Extract entities as nodes ---
            # Pass the full episode object so helper functions can access its properties (like valid_at for created_at)
            # NOTE: Assumes `extract_nodes` (and functions it calls) in `node_operations.py` is modified
            #       to accept the `episode` object and use `episode.valid_at` when setting `created_at` for NEW nodes.
            extracted_nodes = await extract_nodes(
                self.llm_client, episode, previous_episodes, entity_types
            )
            logger.debug(f'Extracted {len(extracted_nodes)} potential nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            # --- 2. Calculate Embeddings for new nodes ---
            await semaphore_gather(
                *[node.generate_name_embedding(self.embedder) for node in extracted_nodes if node.name_embedding is None]
            )

            # --- 3. Find potentially relevant existing nodes ---
            existing_nodes_lists: list[list[EntityNode]] = list(
                await semaphore_gather(
                    *[
                        get_relevant_nodes(self.driver, SearchFilters(), [node])
                        for node in extracted_nodes
                    ]
                )
            )

            # --- 4. Resolve extracted nodes against existing ones & generate summaries/attributes ---
            # NOTE: Assumes `resolve_extracted_nodes` (and `resolve_extracted_node`) in `node_operations.py` is modified
            #       to accept the `episode` object and use `episode.valid_at` when setting `created_at` for NEW nodes created during resolution (e.g., dynamic events).
            (
                resolved_nodes,
                uuid_map,
                new_edges_from_resolution # e.g., PREVIOUS_INSTANCE edges
            ) = await resolve_extracted_nodes(
                self.llm_client,
                extracted_nodes,
                existing_nodes_lists,
                episode, # Pass episode context
                previous_episodes,
                entity_types,
            )
            all_edges_to_save.extend(new_edges_from_resolution)
            logger.debug(f'Resolved to {len(resolved_nodes)} nodes after deduplication/creation.')
            logger.debug(f'UUID map from node resolution: {uuid_map}')
            if new_edges_from_resolution:
                 logger.debug(f'Created {len(new_edges_from_resolution)} new edges during node resolution.')

            # --- 5. Extract potential edges between resolved nodes ---
            # NOTE: Assumes `extract_edges` in `edge_operations.py` is modified
            #       to accept the `episode` object and use `episode.valid_at` when setting `created_at` for NEW edges.
            extracted_edges = await extract_edges(
                self.llm_client, episode, resolved_nodes, previous_episodes, current_group_id
            )
            logger.debug(f'Extracted {len(extracted_edges)} potential edges.')

            # --- 6. Resolve edge pointers based on node deduplication map ---
            extracted_edges_with_resolved_pointers = resolve_edge_pointers(
                extracted_edges, uuid_map
            )

            # --- 7. Calculate embeddings for new edges ---
            await semaphore_gather(
                *[
                    edge.generate_embedding(self.embedder)
                    for edge in extracted_edges_with_resolved_pointers if edge.fact_embedding is None
                ]
            )

            # --- 8. Find relevant existing edges for deduplication and temporal resolution ---
            related_edges_lists: list[list[EntityEdge]] = list(
                await semaphore_gather(
                    *[
                        get_relevant_edges(
                            self.driver,
                            [edge],
                            edge.source_node_uuid,
                            edge.target_node_uuid,
                            RELEVANT_SCHEMA_LIMIT,
                        )
                        for edge in extracted_edges_with_resolved_pointers
                    ]
                )
            )
            # Fetch existing edges connected to the source/target nodes for contradiction checks
            involved_node_uuids = {edge.source_node_uuid for edge in extracted_edges_with_resolved_pointers} | \
                                  {edge.target_node_uuid for edge in extracted_edges_with_resolved_pointers}
            existing_edges_for_contradiction: List[EntityEdge] = []
            if involved_node_uuids:
                 edges_connected_to_involved_nodes = list(await semaphore_gather(
                      *[EntityEdge.get_by_node_uuid(self.driver, node_uuid) for node_uuid in involved_node_uuids]
                 ))
                 existing_edges_for_contradiction_map = {edge.uuid: edge for edge_list in edges_connected_to_involved_nodes for edge in edge_list}
                 existing_edges_for_contradiction = list(existing_edges_for_contradiction_map.values())
            existing_edges_lists_for_contradiction = [existing_edges_for_contradiction] * len(extracted_edges_with_resolved_pointers)


            # --- 9. Resolve extracted edges against existing ones ---
            # NOTE: Assumes `resolve_extracted_edges` in `edge_operations.py` is modified
            #       to accept the `episode` object and use `episode.valid_at` when setting `created_at` for NEW edges created during resolution.
            resolved_edges_final, invalidated_edges = await resolve_extracted_edges(
                self.llm_client,
                extracted_edges_with_resolved_pointers,
                related_edges_lists,
                existing_edges_lists_for_contradiction,
                episode, # Pass episode context
                previous_episodes,
            )
            all_edges_to_save.extend(resolved_edges_final)
            all_edges_to_save.extend(invalidated_edges)
            logger.debug(f'Resolved to {len(resolved_edges_final)} final edges after deduplication.')
            logger.debug(f'Identified {len(invalidated_edges)} edges to mark as invalidated/expired.')


            # --- 10. Build Episodic Edges ---
            # Pass the original message time (`episode.valid_at`) to be used as `created_at` for these edges
            episodic_edges: list[EpisodicEdge] = build_episodic_edges(
                resolved_nodes,
                episode,
                episode.valid_at # Use original timestamp for creation
            )
            logger.debug(f'Built {len(episodic_edges)} episodic edges linking episode to nodes.')

            # --- 11. Link episode to the final set of entity edges created/updated from it ---
            final_edge_uuids_for_episode = {edge.uuid for edge in resolved_edges_final if episode.uuid in edge.episodes} | \
                                           {edge.uuid for edge in new_edges_from_resolution if episode.uuid in edge.episodes}
            episode.entity_edges = sorted(list(final_edge_uuids_for_episode))

            # --- 12. Optionally clear raw content ---
            if not self.store_raw_episode_content:
                episode.content = ''

            # --- 13. Save all nodes and edges in bulk ---
            # Ensure resolved_nodes contains nodes with correct created_at if they were newly created
            # Ensure all_edges_to_save contains edges with correct created_at if they were newly created
            final_entity_edges_to_save_map = {edge.uuid: edge for edge in all_edges_to_save}

            # Ensure all nodes and edges have created_at set before saving
            for node in resolved_nodes:
                if node.created_at is None:
                    logger.warning(f"Node {node.uuid} missing created_at before save, defaulting to episode time.")
                    node.created_at = episode.valid_at
            for edge_obj in final_entity_edges_to_save_map.values():
                 if edge_obj.created_at is None:
                    logger.warning(f"Edge {edge_obj.uuid} missing created_at before save, defaulting to episode time.")
                    edge_obj.created_at = episode.valid_at

            await add_nodes_and_edges_bulk(
                self.driver,
                [episode],
                episodic_edges,
                resolved_nodes,
                list(final_entity_edges_to_save_map.values())
            )
            logger.info(f"Bulk saved: 1 episode, {len(episodic_edges)} episodic edges, {len(resolved_nodes)} entity nodes, {len(final_entity_edges_to_save_map)} entity edges.")

            # --- 14. Update communities if requested ---
            if update_communities:
                await semaphore_gather(
                    *[
                        update_community(self.driver, self.llm_client, self.embedder, node)
                        for node in resolved_nodes
                    ]
                )
                logger.debug("Community update process triggered.")

            end_time_processing = time()
            logger.info(f'Completed add_episode for episode {episode.uuid} in {(end_time_processing - start_time_processing) * 1000:.2f} ms')

            # Return the final state
            return AddEpisodeResults(
                 episode=episode,
                 nodes=resolved_nodes,
                 edges=list(final_entity_edges_to_save_map.values())
            )

        except Exception as e:
            logger.error(f"Error during add_episode for {uuid or 'new episode'}: {e}", exc_info=True)
            raise # Re-raise the exception

    async def build_communities(self, group_ids: list[str] | None = None) -> list[CommunityNode]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        group_ids : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        logger.info(f"Starting community building for group_ids: {group_ids or 'all'}")
        await remove_communities(self.driver)
        logger.debug("Cleared existing communities.")

        community_nodes, community_edges = await build_communities(
            self.driver, self.llm_client, group_ids
        )
        logger.info(f"Identified {len(community_nodes)} potential communities.")

        # Generate embeddings for new community nodes
        await semaphore_gather(
            *[node.generate_name_embedding(self.embedder) for node in community_nodes if node.name_embedding is None]
        )
        logger.debug("Generated embeddings for community nodes.")

        # Save new community nodes and edges
        # Ensure created_at is set (should be handled by build_communities if modified)
        for node in community_nodes:
            if node.created_at is None: node.created_at = utc_now() # Fallback
        for edge in community_edges:
             if edge.created_at is None: edge.created_at = utc_now() # Fallback

        await semaphore_gather(*[node.save(self.driver) for node in community_nodes])
        await semaphore_gather(*[edge.save(self.driver) for edge in community_edges])
        logger.info(f"Saved {len(community_nodes)} community nodes and {len(community_edges)} community edges.")

        return community_nodes

    async def search(
        self,
        query: str,
        center_node_uuid: str | None = None,
        group_ids: list[str] | None = None,
        num_results=DEFAULT_SEARCH_LIMIT,
        search_filter: SearchFilters | None = None,
    ) -> list[EntityEdge]:
        """
        Perform a hybrid search on the knowledge graph for relevant facts (edges).
        """
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        # Define priority channels (consider making this configurable)
        priority_channels = {
            "#whats-new-product", "#product-x-business", "#pods-sourcing", "#announcement-x-support"
        }

        search_results: SearchResults = await self._search(
                query=query,
                config=search_config,
                group_ids=group_ids if group_ids else None,
                center_node_uuid=center_node_uuid,
                search_filter=search_filter if search_filter is not None else SearchFilters(),
                priority_channel_ids=list(priority_channels), # Pass priority info
                boost_human=True, # Example: Boost human messages
                priority_boost=0.2 # Example boost factor
            )

        return search_results.edges

    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
        # --- MODIFICATION START: Accept priority parameters ---
        priority_channel_ids: Optional[List[str]] = None,
        boost_human: bool = True,
        priority_boost: float = 0.2,
        # --- MODIFICATION END ---
    ) -> SearchResults:
        """Internal search method allowing full SearchConfig control."""
        # Pass priority info down to the actual search implementation
        return await search( # Call the imported search function
            driver=self.driver,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            query=query,
            group_ids=group_ids,
            config=config,
            search_filter=search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
             # --- MODIFICATION START ---
            priority_channel_ids=priority_channel_ids,
            boost_human=boost_human,
            priority_boost=priority_boost
             # --- MODIFICATION END ---
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        """Retrieve nodes and edges directly associated with specific episode UUIDs."""
        if not episode_uuids:
             return SearchResults(edges=[], nodes=[], communities=[])

        try:
            episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)
        except NodeNotFoundError:
             logger.warning(f"One or more episode UUIDs not found: {episode_uuids}")
             episodes = []

        if not episodes:
            return SearchResults(edges=[], nodes=[], communities=[])

        all_entity_edge_uuids = {uuid for ep in episodes for uuid in ep.entity_edges}

        edges: list[EntityEdge] = []
        if all_entity_edge_uuids:
             try:
                 edges = await EntityEdge.get_by_uuids(self.driver, list(all_entity_edge_uuids))
             except EdgeNotFoundError:
                  logger.warning(f"One or more entity edge UUIDs linked from episodes {episode_uuids} not found.")

        nodes = await get_mentioned_nodes(self.driver, episodes)

        return SearchResults(edges=edges, nodes=nodes, communities=[])

    async def add_triplet(self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode):
        """Manually add a single fact (triplet) to the graph, performing necessary checks."""
        logger.info(f"Attempting to add triplet: {source_node.name} -[{edge.name}]-> {target_node.name}")

        # Use current time for manually added triplets unless specified
        manual_add_time = utc_now()
        if source_node.created_at is None: source_node.created_at = manual_add_time
        if target_node.created_at is None: target_node.created_at = manual_add_time
        if edge.created_at is None: edge.created_at = manual_add_time
        if edge.valid_at is None: edge.valid_at = manual_add_time # Default validity to creation time

        if source_node.name_embedding is None: await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None: await target_node.generate_name_embedding(self.embedder)
        if edge.fact_embedding is None: await edge.generate_embedding(self.embedder)

        # --- Resolve source and target nodes ---
        # Pass a dummy episode object or None, and ensure resolve_extracted_node handles it
        # It needs the timestamp mainly for setting created_at on *new* nodes during resolution
        dummy_episode_context = EpisodicNode( # Create a minimal context
             name="ManualTriplet", source=EpisodeType.text, content="", source_description="Manual Add",
             valid_at=manual_add_time, created_at=manual_add_time, group_id=source_node.group_id # Use consistent group
        )
        resolved_source_tuple = await resolve_extracted_node(
            self.llm_client, source_node,
            await get_relevant_nodes(self.driver, SearchFilters(), [source_node]),
            dummy_episode_context # Pass context
        )
        resolved_target_tuple = await resolve_extracted_node(
            self.llm_client, target_node,
            await get_relevant_nodes(self.driver, SearchFilters(), [target_node]),
             dummy_episode_context # Pass context
        )

        resolved_source_node = resolved_source_tuple[0]
        resolved_target_node = resolved_target_tuple[0]
        new_edges_from_nodes = [e for e in [resolved_source_tuple[2], resolved_target_tuple[2]] if e]

        edge.source_node_uuid = resolved_source_node.uuid
        edge.target_node_uuid = resolved_target_node.uuid
        if not edge.episodes: edge.episodes = []

        # --- Deduplicate and resolve contradictions ---
        related_edges = await get_relevant_edges(
            self.driver, [edge], resolved_source_node.uuid, resolved_target_node.uuid,
        )
        # Pass dummy episode context if needed by dedupe/contradiction logic
        resolved_edge = await dedupe_extracted_edge(self.llm_client, edge, related_edges)

        existing_edges_for_contradiction_s = await EntityEdge.get_by_node_uuid(self.driver, resolved_source_node.uuid)
        existing_edges_for_contradiction_t = await EntityEdge.get_by_node_uuid(self.driver, resolved_target_node.uuid)
        existing_edges_map = {e.uuid: e for e in existing_edges_for_contradiction_s + existing_edges_for_contradiction_t}

        contradicting_edges = await get_edge_contradictions(self.llm_client, resolved_edge, list(existing_edges_map.values()))
        invalidated_edges = resolve_edge_contradictions(resolved_edge, contradicting_edges) # This function only updates expiry, doesn't need episode time

        # --- Save ---
        nodes_to_save = [resolved_source_node, resolved_target_node]
        edges_to_save = [resolved_edge] + invalidated_edges + new_edges_from_nodes
        final_edges_map = {e.uuid: e for e in edges_to_save}

        # Ensure created_at is set before saving
        for node in nodes_to_save:
            if node.created_at is None: node.created_at = manual_add_time
        for edge_obj in final_edges_map.values():
            if edge_obj.created_at is None: edge_obj.created_at = manual_add_time


        await add_nodes_and_edges_bulk(
            self.driver, [], [], nodes_to_save, list(final_edges_map.values())
        )
        logger.info(f"Successfully added/updated triplet: {resolved_source_node.name} -[{resolved_edge.name}]-> {resolved_target_node.name}")

    async def remove_episode(self, episode_uuid: str):
        """
        Removes an episode and attempts to clean up nodes/edges uniquely created by it.
        Caution: Determining unique creation can be complex.
        """
        logger.warning(f"Attempting to remove episode {episode_uuid} and potentially related data.")
        try:
            episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

            edges_to_potentially_delete: list[EntityEdge] = []
            if episode.entity_edges:
                 try:
                     mentioned_edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)
                     for edge in mentioned_edges:
                         if len(edge.episodes) == 1 and edge.episodes[0] == episode.uuid:
                             edges_to_potentially_delete.append(edge)
                 except EdgeNotFoundError:
                      logger.warning(f"Some edges listed in episode {episode_uuid} not found.")

            nodes_mentioned = await get_mentioned_nodes(self.driver, [episode])
            nodes_to_potentially_delete: list[EntityNode] = []
            for node in nodes_mentioned:
                query_ep: LiteralString = """
                    MATCH (ep:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid})
                    WHERE ep.uuid <> $episode_uuid
                    RETURN count(ep) AS other_episode_count
                """
                records_ep, _, _ = await self.driver.execute_query(
                    query_ep, uuid=node.uuid, episode_uuid=episode_uuid, database_=DEFAULT_DATABASE, routing_='r'
                )
                other_mentions = records_ep[0]['other_episode_count'] if records_ep else 0

                if other_mentions == 0:
                    query_edges: LiteralString = """
                         MATCH (n:Entity {uuid: $uuid})-[r:RELATES_TO]-()
                         WHERE NOT r.uuid IN $edges_being_deleted_uuids
                         RETURN count(r) as edge_count
                    """
                    edge_records, _, _ = await self.driver.execute_query(
                         query_edges,
                         uuid=node.uuid,
                         edges_being_deleted_uuids=[e.uuid for e in edges_to_potentially_delete],
                         database_=DEFAULT_DATABASE, routing_='r'
                    )
                    remaining_edges = edge_records[0]['edge_count'] if edge_records else 0
                    if remaining_edges == 0:
                         nodes_to_potentially_delete.append(node)

            logger.info(f"Deleting {len(edges_to_potentially_delete)} edges potentially unique to episode {episode_uuid}.")
            await semaphore_gather(*[edge.delete(self.driver) for edge in edges_to_potentially_delete])

            logger.info(f"Deleting {len(nodes_to_potentially_delete)} nodes potentially unique to episode {episode_uuid}.")
            await semaphore_gather(*[node.delete(self.driver) for node in nodes_to_potentially_delete])

            logger.info(f"Deleting episode node {episode_uuid}.")
            await episode.delete(self.driver)

            logger.warning(f"Completed removal process for episode {episode_uuid}.")

        except NodeNotFoundError:
            raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.") from None
        except Exception as e:
             logger.error(f"Error removing episode {episode_uuid}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Error removing episode {episode_uuid}")
        