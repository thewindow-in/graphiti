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

MODIFIED TO HANDLE EDGES CREATED DURING NODE RESOLUTION
"""

from http.client import HTTPException
import logging
from datetime import datetime
from time import time
from typing import List, Optional # Added List, Optional

from dotenv import load_dotenv
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from neo4j import AsyncGraphDatabase
from pydantic import BaseModel
from typing_extensions import LiteralString

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.embedder import EmbedderClient, OpenAIEmbedder
from graphiti_core.helpers import DEFAULT_DATABASE, semaphore_gather
from graphiti_core.llm_client import LLMClient, OpenAIClient
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode
from graphiti_core.search.search import SearchConfig, search
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
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    extract_edge_dates_bulk,
    extract_nodes_and_edges_bulk,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
)
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.community_operations import (
    build_communities,
    remove_communities,
    update_community,
)
from graphiti_core.utils.maintenance.edge_operations import (
    build_episodic_edges,
    dedupe_extracted_edge,
    extract_edges,
    resolve_edge_contradictions,
    resolve_extracted_edges,
)
from graphiti_core.utils.maintenance.graph_data_operations import (
    EPISODE_WINDOW_LEN,
    build_indices_and_constraints,
    retrieve_episodes,
)
from graphiti_core.utils.maintenance.node_operations import (
    extract_nodes,
    resolve_extracted_node,
    resolve_extracted_nodes, # Keep this import
)
from graphiti_core.utils.maintenance.temporal_operations import get_edge_contradictions

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
        # Ensure group_ids is None if empty list is passed
        effective_group_ids = group_ids if group_ids else None
        return await retrieve_episodes(self.driver, reference_time, last_n, effective_group_ids)

    async def add_episode(
        self,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType = EpisodeType.message,
        group_id: str = '',
        uuid: str | None = None,
        update_communities: bool = False,
        entity_types: dict[str, BaseModel] | None = None,
    ) -> AddEpisodeResults:
        """
        Process an episode and update the graph.
        Handles node/edge extraction, deduplication, embedding, temporal resolution,
        and saving to the graph. Includes logic for handling recurring dynamic events.
        """
        try:
            start = time()
            all_edges_to_save: list[EntityEdge] = [] # Collect all entity edges here
            now = utc_now()

            # Ensure group_id is treated consistently (e.g., handle empty string if needed)
            current_group_id = group_id or 'default' # Example: use 'default' if empty

            previous_episodes = await self.retrieve_episodes(
                reference_time, last_n=RELEVANT_SCHEMA_LIMIT, group_ids=[current_group_id]
            )

            # Create or retrieve the episodic node
            if uuid:
                 try:
                     episode = await EpisodicNode.get_by_uuid(self.driver, uuid)
                     # Optionally update existing episode fields if needed
                     episode.content = episode_body # Example update
                     episode.valid_at = reference_time
                 except NodeNotFoundError:
                      logger.warning(f"Episode UUID {uuid} provided but not found. Creating new episode.")
                      uuid = None # Fallback to creating a new one
            if not uuid: # Create new if no UUID or if lookup failed
                episode = EpisodicNode(
                    name=name,
                    group_id=current_group_id,
                    labels=[], # Labels for EpisodicNode? Usually not needed.
                    source=source,
                    content=episode_body,
                    source_description=source_description,
                    created_at=now,
                    valid_at=reference_time,
                )

            # 1. Extract entities as nodes
            extracted_nodes = await extract_nodes(
                self.llm_client, episode, previous_episodes, entity_types
            )
            logger.debug(f'Extracted {len(extracted_nodes)} potential nodes: {[(n.name, n.uuid) for n in extracted_nodes]}')

            # 2. Calculate Embeddings for new nodes
            await semaphore_gather(
                *[node.generate_name_embedding(self.embedder) for node in extracted_nodes if node.name_embedding is None]
            )

            # 3. Find potentially relevant existing nodes
            # Use a set to avoid fetching duplicates if extracted nodes have same name/group
            unique_group_ids = {node.group_id for node in extracted_nodes}
            # Fetch relevant nodes based on *all* extracted nodes for efficiency?
            # Or fetch per node as before? Per-node is safer for context relevance.
            existing_nodes_lists: list[list[EntityNode]] = list(
                await semaphore_gather(
                    *[
                        get_relevant_nodes(self.driver, SearchFilters(), [node]) # Pass node list
                        for node in extracted_nodes
                    ]
                )
            )

            # 4. Resolve extracted nodes against existing ones & generate summaries/attributes
            # <<< CHANGE START >>>
            # resolve_extracted_nodes now returns new_edges_from_resolution
            (
                resolved_nodes,
                uuid_map,
                new_edges_from_resolution # e.g., PREVIOUS_INSTANCE edges
            ) = await resolve_extracted_nodes(
                self.llm_client,
                extracted_nodes,
                existing_nodes_lists,
                episode,
                previous_episodes,
                entity_types,
            )
            all_edges_to_save.extend(new_edges_from_resolution) # Add these to be saved
            logger.debug(f'Resolved to {len(resolved_nodes)} nodes after deduplication/creation.')
            logger.debug(f'UUID map from node resolution: {uuid_map}')
            if new_edges_from_resolution:
                 logger.debug(f'Created {len(new_edges_from_resolution)} new edges during node resolution (e.g., PREVIOUS_INSTANCE).')
            # <<< CHANGE END >>>

            # 5. Extract potential edges between resolved nodes based on the current episode
            # Pass resolved_nodes which might include existing nodes with summaries
            extracted_edges = await extract_edges(
                self.llm_client, episode, resolved_nodes, previous_episodes, current_group_id
            )
            logger.debug(f'Extracted {len(extracted_edges)} potential edges.')

            # 6. Resolve edge pointers based on node deduplication map
            # Ensure this uses the final uuid_map from resolve_extracted_nodes
            extracted_edges_with_resolved_pointers = resolve_edge_pointers(
                extracted_edges, uuid_map
            )

            # 7. Calculate embeddings for new edges
            await semaphore_gather(
                *[
                    edge.generate_embedding(self.embedder)
                    for edge in extracted_edges_with_resolved_pointers if edge.fact_embedding is None
                ]
            )

            # 8. Find relevant existing edges for deduplication and temporal resolution
            # Fetch related edges based on the source/target of the *potentially new* edges
            related_edges_lists: list[list[EntityEdge]] = list(
                await semaphore_gather(
                    *[
                        get_relevant_edges(
                            self.driver,
                            [edge], # Check relevance for this specific edge
                            edge.source_node_uuid,
                            edge.target_node_uuid,
                            RELEVANT_SCHEMA_LIMIT,
                        )
                        for edge in extracted_edges_with_resolved_pointers
                    ]
                )
            )
            # Fetch existing edges connected to the source/target nodes for contradiction checks
            # Combine fetches for efficiency
            involved_node_uuids = {edge.source_node_uuid for edge in extracted_edges_with_resolved_pointers} | \
                                  {edge.target_node_uuid for edge in extracted_edges_with_resolved_pointers}

            # Fetch edges connected to any involved node (simpler approach)
            # This might fetch more than needed but reduces complexity
            existing_edges_for_contradiction: List[EntityEdge] = []
            if involved_node_uuids:
                 # Using EntityEdge.get_by_node_uuid might be simpler if available and efficient
                 # Otherwise, construct a query
                 # Note: This fetches edges connected to *any* involved node.
                 # The original logic fetched per-edge source/target separately.
                 # This simpler approach might be less precise for contradiction context but faster.
                 # Adjust if needed.
                 edges_connected_to_involved_nodes = list(await semaphore_gather(
                      *[EntityEdge.get_by_node_uuid(self.driver, node_uuid) for node_uuid in involved_node_uuids]
                 ))
                 # Flatten and deduplicate
                 existing_edges_for_contradiction_map = {edge.uuid: edge for edge_list in edges_connected_to_involved_nodes for edge in edge_list}
                 existing_edges_for_contradiction = list(existing_edges_for_contradiction_map.values())


            # Assign the same comprehensive list for contradiction checks for all edges
            # This simplifies the structure compared to the original per-edge source/target lists
            existing_edges_lists_for_contradiction = [existing_edges_for_contradiction] * len(extracted_edges_with_resolved_pointers)


            # 9. Resolve extracted edges against existing ones (deduplication, temporal analysis)
            resolved_edges_final, invalidated_edges = await resolve_extracted_edges(
                self.llm_client,
                extracted_edges_with_resolved_pointers,
                related_edges_lists, # For deduplication context
                existing_edges_lists_for_contradiction, # For contradiction context
                episode,
                previous_episodes,
            )
            all_edges_to_save.extend(resolved_edges_final)
            all_edges_to_save.extend(invalidated_edges) # Add invalidated edges (they need saving with updated expiry)
            logger.debug(f'Resolved to {len(resolved_edges_final)} final edges after deduplication.')
            logger.debug(f'Identified {len(invalidated_edges)} edges to mark as invalidated/expired.')


            # 10. Build Episodic Edges (linking episode to resolved nodes)
            # Use resolved_nodes which contains the final set of nodes (new or existing) mentioned
            episodic_edges: list[EpisodicEdge] = build_episodic_edges(resolved_nodes, episode, now)
            logger.debug(f'Built {len(episodic_edges)} episodic edges linking episode to nodes.')

            # 11. Link episode to the final set of entity edges created/updated from it
            # Filter edges that were genuinely created or updated *from this episode*
            # This includes newly created edges and existing edges that had this episode added to their list
            final_edge_uuids_for_episode = {edge.uuid for edge in resolved_edges_final if episode.uuid in edge.episodes} | \
                                           {edge.uuid for edge in new_edges_from_resolution if episode.uuid in edge.episodes} # Include PREVIOUS_INSTANCE if linked

            episode.entity_edges = sorted(list(final_edge_uuids_for_episode))

            # 12. Optionally clear raw content
            if not self.store_raw_episode_content:
                episode.content = ''

            # 13. Save all nodes and edges in bulk
            # Ensure we save the final state of resolved_nodes (might include updated existing nodes)
            # Ensure we save all collected entity edges (new, updated existing, invalidated, PREVIOUS_INSTANCE)
            # Deduplicate edges before saving just in case
            final_entity_edges_to_save_map = {edge.uuid: edge for edge in all_edges_to_save}

            await add_nodes_and_edges_bulk(
                self.driver,
                [episode], # Save the episode node (potentially updated)
                episodic_edges,
                resolved_nodes, # Save the final state of involved nodes
                list(final_entity_edges_to_save_map.values()) # Save unique entity edges
            )
            logger.info(f"Bulk saved: 1 episode, {len(episodic_edges)} episodic edges, {len(resolved_nodes)} entity nodes, {len(final_entity_edges_to_save_map)} entity edges.")

            # 14. Update communities if requested
            if update_communities:
                # Update communities based on the nodes that were actually involved/modified
                await semaphore_gather(
                    *[
                        update_community(self.driver, self.llm_client, self.embedder, node)
                        for node in resolved_nodes # Use the final list of nodes
                    ]
                )
                logger.debug("Community update process triggered.")

            end = time()
            logger.info(f'Completed add_episode for episode {episode.uuid} in {(end - start) * 1000:.2f} ms')

            # Return the final state
            return AddEpisodeResults(
                 episode=episode,
                 nodes=resolved_nodes,
                 edges=list(final_entity_edges_to_save_map.values())
            )

        except Exception as e:
            logger.error(f"Error during add_episode: {e}", exc_info=True)
            # Re-raise the exception so the background worker can log it
            raise # Keep this to ensure errors propagate

    async def build_communities(self, group_ids: list[str] | None = None) -> list[CommunityNode]:
        """
        Use a community clustering algorithm to find communities of nodes. Create community nodes summarising
        the content of these communities.
        ----------
        group_ids : list[str] | None
            Optional. Create communities only for the listed group_ids. If blank the entire graph will be used.
        """
        logger.info(f"Starting community building for group_ids: {group_ids or 'all'}")
        # Clear existing communities before rebuilding
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
        # Default to RRF search if no center node, otherwise use node distance reranking
        search_config = (
            EDGE_HYBRID_SEARCH_RRF if center_node_uuid is None else EDGE_HYBRID_SEARCH_NODE_DISTANCE
        )
        search_config.limit = num_results

        # Use the internal _search method
        search_results: SearchResults = await self._search(
                query=query,
                config=search_config,
                group_ids=group_ids if group_ids else None, # Handle empty list
                center_node_uuid=center_node_uuid,
                search_filter=search_filter if search_filter is not None else SearchFilters(),
                # bfs_origin_node_uuids=None # BFS origin usually not needed for simple fact search
            )

        return search_results.edges # Return only the edges as per original signature

    async def _search(
        self,
        query: str,
        config: SearchConfig,
        group_ids: list[str] | None = None,
        center_node_uuid: str | None = None,
        bfs_origin_node_uuids: list[str] | None = None,
        search_filter: SearchFilters | None = None,
    ) -> SearchResults:
        """Internal search method allowing full SearchConfig control."""
        return await search(
            driver=self.driver,
            embedder=self.embedder,
            cross_encoder=self.cross_encoder,
            query=query,
            group_ids=group_ids, # Pass along potentially None value
            config=config,
            search_filter=search_filter if search_filter is not None else SearchFilters(),
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
        )

    async def get_nodes_and_edges_by_episode(self, episode_uuids: list[str]) -> SearchResults:
        """Retrieve nodes and edges directly associated with specific episode UUIDs."""
        if not episode_uuids:
             return SearchResults(edges=[], nodes=[], communities=[])

        # Fetch episodes first to get their associated entity edge UUIDs
        try:
            episodes = await EpisodicNode.get_by_uuids(self.driver, episode_uuids)
        except NodeNotFoundError:
             # Handle case where some/all episode UUIDs might be invalid
             logger.warning(f"One or more episode UUIDs not found: {episode_uuids}")
             episodes = [] # Or filter out invalid ones if partial results are okay

        if not episodes:
            return SearchResults(edges=[], nodes=[], communities=[])

        # Collect all unique entity edge UUIDs mentioned across the found episodes
        all_entity_edge_uuids = {uuid for ep in episodes for uuid in ep.entity_edges}

        # Fetch the actual entity edges
        edges: list[EntityEdge] = []
        if all_entity_edge_uuids:
             try:
                 # Fetch in potentially multiple batches if list is very large
                 edges = await EntityEdge.get_by_uuids(self.driver, list(all_entity_edge_uuids))
             except EdgeNotFoundError:
                  logger.warning(f"One or more entity edge UUIDs linked from episodes {episode_uuids} not found.")
                  # Continue with edges that were found

        # Fetch nodes mentioned by these episodes
        # get_mentioned_nodes likely needs to handle the case where episodes list might be empty
        nodes = await get_mentioned_nodes(self.driver, episodes)

        # Communities are not directly linked to episodes in this way
        return SearchResults(edges=edges, nodes=nodes, communities=[])

    # This method seems more like a maintenance/manual addition, less part of automated flow
    async def add_triplet(self, source_node: EntityNode, edge: EntityEdge, target_node: EntityNode):
        """Manually add a single fact (triplet) to the graph, performing necessary checks."""
        logger.info(f"Attempting to add triplet: {source_node.name} -[{edge.name}]-> {target_node.name}")

        # Ensure nodes have embeddings
        if source_node.name_embedding is None:
            await source_node.generate_name_embedding(self.embedder)
        if target_node.name_embedding is None:
            await target_node.generate_name_embedding(self.embedder)
        # Ensure edge has embedding
        if edge.fact_embedding is None:
            await edge.generate_embedding(self.embedder)

        # Resolve source and target nodes against existing graph data
        # Note: This uses the more complex resolve_extracted_node which also generates summaries
        # Might be overkill if nodes are assumed to exist or have summaries already.
        # Consider a simpler "find or create" if performance is key here.
        resolved_source_tuple = await resolve_extracted_node(
            self.llm_client,
            source_node,
            await get_relevant_nodes(self.driver, SearchFilters(), [source_node]),
            # No episode context for manual add
        )
        resolved_target_tuple = await resolve_extracted_node(
            self.llm_client,
            target_node,
            await get_relevant_nodes(self.driver, SearchFilters(), [target_node]),
             # No episode context for manual add
        )

        resolved_source_node = resolved_source_tuple[0]
        resolved_target_node = resolved_target_tuple[0]
        # Collect uuid maps if needed, though less critical for single triplet add
        uuid_map = {**resolved_source_tuple[1], **resolved_target_tuple[1]}
        # Collect potential PREVIOUS_INSTANCE edges (unlikely for manual add, but handle)
        new_edges_from_nodes = [e for e in [resolved_source_tuple[2], resolved_target_tuple[2]] if e]


        # Update edge pointers based on resolved nodes
        edge.source_node_uuid = resolved_source_node.uuid
        edge.target_node_uuid = resolved_target_node.uuid
        # Manually added edges usually don't belong to an episode unless specified
        if not edge.episodes:
             edge.episodes = [] # Ensure it's an empty list, not None

        # Deduplicate the edge against existing edges between the resolved nodes
        related_edges = await get_relevant_edges(
            self.driver,
            [edge], # Check relevance for this specific edge
            resolved_source_node.uuid,
            resolved_target_node.uuid,
        )
        resolved_edge = await dedupe_extracted_edge(self.llm_client, edge, related_edges)

        # Check for contradictions (temporal or factual)
        # Fetch existing edges connected to either node for contradiction check
        existing_edges_for_contradiction_s = await EntityEdge.get_by_node_uuid(self.driver, resolved_source_node.uuid)
        existing_edges_for_contradiction_t = await EntityEdge.get_by_node_uuid(self.driver, resolved_target_node.uuid)
        existing_edges_map = {e.uuid: e for e in existing_edges_for_contradiction_s + existing_edges_for_contradiction_t}

        contradicting_edges = await get_edge_contradictions(self.llm_client, resolved_edge, list(existing_edges_map.values()))
        invalidated_edges = resolve_edge_contradictions(resolved_edge, contradicting_edges)

        # Save all affected nodes and edges
        nodes_to_save = [resolved_source_node, resolved_target_node]
        edges_to_save = [resolved_edge] + invalidated_edges + new_edges_from_nodes
        # Deduplicate edges before saving
        final_edges_map = {e.uuid: e for e in edges_to_save}

        await add_nodes_and_edges_bulk(
            self.driver,
            [], # No episodic nodes
            [], # No episodic edges
            nodes_to_save,
            list(final_edges_map.values())
        )
        logger.info(f"Successfully added/updated triplet and related elements for: {resolved_source_node.name} -[{resolved_edge.name}]-> {resolved_target_node.name}")


    async def remove_episode(self, episode_uuid: str):
        """
        Removes an episode and attempts to clean up nodes/edges uniquely created by it.
        Caution: Determining unique creation can be complex.
        """
        logger.warning(f"Attempting to remove episode {episode_uuid} and potentially related data.")
        try:
            # Find the episode to be deleted
            episode = await EpisodicNode.get_by_uuid(self.driver, episode_uuid)

            # Find entity edges potentially linked *only* to this episode
            # This is heuristic: assumes first episode in list is the creator. Risky.
            edges_to_potentially_delete: list[EntityEdge] = []
            if episode.entity_edges:
                 try:
                     mentioned_edges = await EntityEdge.get_by_uuids(self.driver, episode.entity_edges)
                     for edge in mentioned_edges:
                         # If this episode is the ONLY one listed, consider deleting edge
                         if len(edge.episodes) == 1 and edge.episodes[0] == episode.uuid:
                             edges_to_potentially_delete.append(edge)
                         # Optional: If this episode is the *first* listed, consider removing it from list
                         # elif edge.episodes and edge.episodes[0] == episode.uuid:
                         #     edge.episodes.pop(0)
                         #     # Need to save the updated edge here if not deleting
                 except EdgeNotFoundError:
                      logger.warning(f"Some edges listed in episode {episode_uuid} not found.")


            # Find nodes mentioned by the episode
            nodes_mentioned = await get_mentioned_nodes(self.driver, [episode])
            nodes_to_potentially_delete: list[EntityNode] = []
            # Check if nodes are mentioned by *other* episodes
            for node in nodes_mentioned:
                # Query to count mentioning episodes *other than* the one being deleted
                query: LiteralString = """
                    MATCH (ep:Episodic)-[:MENTIONS]->(n:Entity {uuid: $uuid})
                    WHERE ep.uuid <> $episode_uuid
                    RETURN count(ep) AS other_episode_count
                """
                records, _, _ = await self.driver.execute_query(
                    query, uuid=node.uuid, episode_uuid=episode_uuid, database_=DEFAULT_DATABASE, routing_='r'
                )
                other_mentions = records[0]['other_episode_count'] if records else 0

                if other_mentions == 0:
                    # Also check if node is involved in edges not being deleted
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

            # Perform deletions
            # It's safer to delete edges first, then nodes, then the episode
            logger.info(f"Deleting {len(edges_to_potentially_delete)} edges potentially unique to episode {episode_uuid}.")
            await semaphore_gather(*[edge.delete(self.driver) for edge in edges_to_potentially_delete])

            logger.info(f"Deleting {len(nodes_to_potentially_delete)} nodes potentially unique to episode {episode_uuid}.")
            await semaphore_gather(*[node.delete(self.driver) for node in nodes_to_potentially_delete])

            logger.info(f"Deleting episode node {episode_uuid}.")
            await episode.delete(self.driver) # Delete the episode itself last

            logger.warning(f"Completed removal process for episode {episode_uuid}.")

        except NodeNotFoundError:
            raise HTTPException(status_code=404, detail=f"Episode {episode_uuid} not found.") from None
        except Exception as e:
             logger.error(f"Error removing episode {episode_uuid}: {e}", exc_info=True)
             raise HTTPException(status_code=500, detail=f"Error removing episode {episode_uuid}")


