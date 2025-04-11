# graphiti_core/search/search.py
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

MODIFIED TO:
- Accept priority parameters.
- Implement post-RRF boosting based on priority/human status.
- Fix Pydantic validation error in edge_search default value.
"""

import logging
from collections import defaultdict
from time import time
# --- MODIFICATION START: Add imports ---
from typing import Optional, List, Dict, Set, Any
from neo4j import AsyncDriver, Query # Add Query
from typing_extensions import LiteralString # Add LiteralString
# --- MODIFICATION END ---

from graphiti_core.cross_encoder.client import CrossEncoderClient
# --- MODIFICATION START: Import EntityEdge ---
from graphiti_core.edges import EntityEdge
# --- MODIFICATION END ---
from graphiti_core.embedder import EmbedderClient
from graphiti_core.errors import SearchRerankerError
from graphiti_core.helpers import semaphore_gather, DEFAULT_DATABASE # Added DEFAULT_DATABASE
from graphiti_core.nodes import CommunityNode, EntityNode
from graphiti_core.search.search_config import (
    DEFAULT_SEARCH_LIMIT,
    CommunityReranker,
    CommunitySearchConfig,
    CommunitySearchMethod, # Added CommunitySearchMethod
    EdgeReranker,
    EdgeSearchConfig,
    EdgeSearchMethod,
    NodeReranker,
    NodeSearchConfig,
    NodeSearchMethod,
    SearchConfig,
    SearchResults,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    community_similarity_search,
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    episode_mentions_reranker,
    maximal_marginal_relevance,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
    rrf,
)

logger = logging.getLogger(__name__)


async def search(
    driver: AsyncDriver,
    embedder: EmbedderClient,
    cross_encoder: CrossEncoderClient,
    query: str,
    group_ids: Optional[List[str]],
    config: SearchConfig,
    search_filter: SearchFilters,
    center_node_uuid: Optional[str] = None,
    bfs_origin_node_uuids: Optional[List[str]] = None,
    # --- MODIFICATION START: Accept priority parameters ---
    priority_channel_ids: Optional[List[str]] = None, # Changed from Set to List for broader compatibility
    boost_human: bool = True,
    priority_boost: float = 0.2,
    # --- MODIFICATION END ---
) -> SearchResults:
    """
    Internal search orchestrator. Fetches query embedding and calls specific search types.
    """
    start = time()
    if query.strip() == '':
        return SearchResults(edges=[], nodes=[], communities=[])

    query_vector = await embedder.create(input_data=[query.replace('\n', ' ')])

    # If group_ids is an empty list, treat it as None for broader matching unless specified otherwise
    effective_group_ids = group_ids if group_ids else None

    # Convert priority channel IDs to a set for efficient lookup
    priority_channels_set = set(priority_channel_ids) if priority_channel_ids else set()

    edges, nodes, communities = await semaphore_gather(
        edge_search(
            driver=driver,
            cross_encoder=cross_encoder,
            query=query,
            query_vector=query_vector,
            group_ids=effective_group_ids,
            config=config.edge_config,
            search_filter=search_filter,
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=config.limit,
            # --- Pass priority info ---
            priority_channels=priority_channels_set,
            boost_human=boost_human,
            priority_boost=priority_boost,
        ),
        node_search(
            driver=driver,
            cross_encoder=cross_encoder,
            query=query,
            query_vector=query_vector,
            group_ids=effective_group_ids,
            config=config.node_config,
            search_filter=search_filter,
            center_node_uuid=center_node_uuid,
            bfs_origin_node_uuids=bfs_origin_node_uuids,
            limit=config.limit,
            # --- Pass priority info ---
            priority_channels=priority_channels_set,
            boost_human=boost_human,
            priority_boost=priority_boost,
        ),
        community_search(
            driver=driver,
            cross_encoder=cross_encoder,
            query=query,
            query_vector=query_vector,
            group_ids=effective_group_ids,
            config=config.community_config,
            limit=config.limit,
            # Note: Community boosting is not implemented here yet
        ),
    )

    results = SearchResults(
        edges=edges,
        nodes=nodes,
        communities=communities,
    )

    latency = (time() - start) * 1000
    logger.debug(f'search returned context for query "{query}" in {latency:.2f} ms')
    return results


async def _fetch_episode_properties(driver: AsyncDriver, episode_uuids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Helper to fetch properties needed for boosting from a list of episode UUIDs."""
    if not episode_uuids:
        return {}

    query_ep_props: LiteralString = """
        UNWIND $episode_uuids AS ep_uuid
        MATCH (ep:Episodic {uuid: ep_uuid})
        RETURN ep.uuid as uuid,
               ep.source_channel_name as channel_name, // Use name or id based on what's stored
               ep.is_human_message as is_human
    """
    try:
        records, _, _ = await driver.execute_query(
            query_ep_props, episode_uuids=episode_uuids, database_=DEFAULT_DATABASE, routing_='r'
        )
        # Check if 'channel_name' or 'is_human' might be null in the database
        return {
            record['uuid']: {
                'channel_name': record.get('channel_name'),
                'is_human': record.get('is_human', True) # Default to human if missing? Decide policy.
            }
            for record in records
        }
    except Exception as e:
        logger.error(f"Failed to fetch episode properties for boosting: {e}", exc_info=True)
        return {}


async def edge_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: Optional[List[str]],
    config: Optional[EdgeSearchConfig],
    search_filter: SearchFilters,
    center_node_uuid: Optional[str] = None,
    bfs_origin_node_uuids: Optional[List[str]] = None,
    limit=DEFAULT_SEARCH_LIMIT,
    # --- MODIFICATION START: Accept priority args ---
    priority_channels: Optional[Set[str]] = None, # Use Set for efficiency
    boost_human: bool = True,
    priority_boost: float = 0.2,
    # --- MODIFICATION END ---
) -> list[EntityEdge]:
    if config is None:
        return []

    # --- Gather initial search results ---
    search_tasks = []
    if EdgeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(edge_fulltext_search(driver, query, search_filter, group_ids, 2 * limit))
    if EdgeSearchMethod.cosine_similarity in config.search_methods:
         search_tasks.append(edge_similarity_search(
            driver, query_vector, None, None, search_filter, group_ids, 2 * limit, config.sim_min_score
        ))
    if EdgeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids: # Only if origins provided initially
         search_tasks.append(edge_bfs_search(
            driver, bfs_origin_node_uuids, config.bfs_max_depth, search_filter, 2 * limit
        ))

    search_results_list: list[list[EntityEdge]] = list(await semaphore_gather(*search_tasks))

    # Add BFS based on initial results if needed and not already done
    if EdgeSearchMethod.bfs in config.search_methods and not bfs_origin_node_uuids:
        source_node_uuids_from_results = {
            edge.source_node_uuid for result in search_results_list for edge in result
        } | {
            edge.target_node_uuid for result in search_results_list for edge in result # Consider target too?
        }
        if source_node_uuids_from_results:
            bfs_results = await edge_bfs_search(
                driver, list(source_node_uuids_from_results), config.bfs_max_depth, search_filter, 2 * limit
            )
            search_results_list.append(bfs_results)

    # --- Deduplicate results ---
    edge_uuid_map: Dict[str, EntityEdge] = {edge.uuid: edge for result in search_results_list for edge in result}

    # --- Reranking Logic ---
    reranked_uuids: list[str] = []
    search_result_uuids_by_method = [[edge.uuid for edge in result] for result in search_results_list]

    if config.reranker == EdgeReranker.rrf or config.reranker == EdgeReranker.episode_mentions or config.reranker == EdgeReranker.node_distance:
        # Base ranking using RRF
        reranked_uuids_base = rrf(search_result_uuids_by_method)

        if config.reranker == EdgeReranker.node_distance:
            if center_node_uuid is None:
                logger.warning('Node Distance reranker selected but no center_node_uuid provided. Falling back to RRF.')
                reranked_uuids = reranked_uuids_base
            else:
                # Perform node distance reranking (implementation depends on node_distance_reranker logic)
                source_to_edge_map = defaultdict(list)
                for uuid in reranked_uuids_base:
                    edge = edge_uuid_map.get(uuid)
                    if edge:
                         source_to_edge_map[edge.source_node_uuid].append(uuid)

                source_uuids_to_rank = list(source_to_edge_map.keys())
                ranked_source_uuids = await node_distance_reranker(driver, source_uuids_to_rank, center_node_uuid)

                reranked_uuids = [edge_uuid for src_uuid in ranked_source_uuids for edge_uuid in source_to_edge_map[src_uuid]]
        else:
            # Simple RRF or base for episode mentions
            reranked_uuids = reranked_uuids_base

        # --- Post-RRF Boosting ---
        if priority_channels or boost_human:
            logger.debug(f"Applying boost to RRF results. Priority Channels: {priority_channels}, Boost Human: {boost_human}")

            # --- FIX START: Handle missing UUIDs and collect episode UUIDs safely ---
            all_episode_uuids: Set[str] = set()
            valid_reranked_uuids_for_boost: List[str] = []
            for uuid in reranked_uuids:
                edge = edge_uuid_map.get(uuid)
                if edge and edge.episodes: # Check if edge exists and has episodes
                    all_episode_uuids.update(edge.episodes)
                    valid_reranked_uuids_for_boost.append(uuid)
                elif edge: # Edge exists but no episodes
                    valid_reranked_uuids_for_boost.append(uuid)
                else:
                    logger.warning(f"UUID {uuid} from reranking not found in edge_uuid_map during boost calculation. Skipping.")
            # --- FIX END ---

            episode_props_map = await _fetch_episode_properties(driver, list(all_episode_uuids))

            boosted_scores: Dict[str, float] = {}
            # Iterate over the UUIDs confirmed to be in the map
            for i, uuid in enumerate(valid_reranked_uuids_for_boost):
                base_score = 1 / (i + 1) # RRF score contribution
                boost = 0.0
                edge = edge_uuid_map.get(uuid) # We know this exists now
                # Check episodes safely
                if edge and edge.episodes:
                    for ep_uuid in edge.episodes:
                        ep_props = episode_props_map.get(ep_uuid)
                        if ep_props:
                            is_priority = priority_channels and ep_props.get('channel_name') in priority_channels
                            is_human = boost_human and ep_props.get('is_human', True)
                            if is_priority or is_human:
                                boost = priority_boost
                                break # Apply boost once per edge
                boosted_scores[uuid] = base_score + boost

            # Add scores for UUIDs that were skipped (assign base score without boost)
            skipped_uuids = set(reranked_uuids) - set(valid_reranked_uuids_for_boost)
            for i, uuid in enumerate(reranked_uuids):
                if uuid in skipped_uuids:
                     boosted_scores[uuid] = 1 / (i + 1) # Assign base score

            # Sort by boosted score
            reranked_uuids = sorted(boosted_scores, key=boosted_scores.get, reverse=True)
            logger.debug(f"Top 5 boosted edge UUIDs: {reranked_uuids[:5]}")


    elif config.reranker == EdgeReranker.mmr:
        candidates = [
            (edge.uuid, edge.fact_embedding or [0.0] * 1024)
            for edge in edge_uuid_map.values() if edge.fact_embedding
        ]
        if candidates:
            reranked_uuids = maximal_marginal_relevance(
                query_vector, candidates, config.mmr_lambda
            )
        else:
            reranked_uuids = []


    elif config.reranker == EdgeReranker.cross_encoder:
        rrf_base_uuids = rrf(search_result_uuids_by_method)
        candidates_for_ce = [edge_uuid_map[uuid] for uuid in rrf_base_uuids if uuid in edge_uuid_map][:limit * 2]

        fact_to_uuid_map = {edge.fact: edge.uuid for edge in candidates_for_ce}
        if fact_to_uuid_map:
            reranked_facts = await cross_encoder.rank(query, list(fact_to_uuid_map.keys()))
            reranked_uuids = [fact_to_uuid_map[fact] for fact, _ in reranked_facts]
        else:
            reranked_uuids = []

    else: # Fallback if reranker not recognized
        logger.warning(f"Unknown edge reranker '{config.reranker}', defaulting to RRF.")
        reranked_uuids = rrf(search_result_uuids_by_method)


    # --- Final edge list construction ---
    final_edges: List[EntityEdge] = []
    seen_uuids = set()
    for uuid in reranked_uuids:
         # --- FIX: Check if uuid exists in map before accessing ---
         if uuid in edge_uuid_map and uuid not in seen_uuids:
             final_edges.append(edge_uuid_map[uuid])
             seen_uuids.add(uuid)
         # --- FIX END ---

    # Apply episode mentions sort if applicable (after primary reranking and boosting)
    if config.reranker == EdgeReranker.episode_mentions:
        final_edges.sort(key=lambda edge: len(edge.episodes), reverse=True)


    logger.info(f"Edge search returned {len(final_edges[:limit])} results using {config.reranker}.")
    return final_edges[:limit]


async def node_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: Optional[List[str]],
    config: Optional[NodeSearchConfig],
    search_filter: SearchFilters,
    center_node_uuid: Optional[str] = None,
    bfs_origin_node_uuids: Optional[List[str]] = None,
    limit=DEFAULT_SEARCH_LIMIT,
     # --- MODIFICATION START: Accept priority args ---
    priority_channels: Optional[Set[str]] = None,
    boost_human: bool = True,
    priority_boost: float = 0.2,
    # --- MODIFICATION END ---
) -> list[EntityNode]:
    if config is None:
        return []

    # --- Gather initial search results ---
    search_tasks = []
    if NodeSearchMethod.bm25 in config.search_methods:
        search_tasks.append(node_fulltext_search(driver, query, search_filter, group_ids, 2 * limit))
    if NodeSearchMethod.cosine_similarity in config.search_methods:
        search_tasks.append(node_similarity_search(
            driver, query_vector, search_filter, group_ids, 2 * limit, config.sim_min_score
        ))
    if NodeSearchMethod.bfs in config.search_methods and bfs_origin_node_uuids:
        search_tasks.append(node_bfs_search(
            driver, bfs_origin_node_uuids, search_filter, config.bfs_max_depth, 2 * limit
        ))

    search_results_list: list[list[EntityNode]] = list(await semaphore_gather(*search_tasks))

    # Add BFS based on initial results if needed
    if NodeSearchMethod.bfs in config.search_methods and not bfs_origin_node_uuids:
        origin_node_uuids_from_results = {
            node.uuid for result in search_results_list for node in result
        }
        if origin_node_uuids_from_results:
            bfs_results = await node_bfs_search(
                driver, list(origin_node_uuids_from_results), search_filter, config.bfs_max_depth, 2 * limit
            )
            search_results_list.append(bfs_results)

    # --- Deduplicate results ---
    node_uuid_map: Dict[str, EntityNode] = {node.uuid: node for result in search_results_list for node in result}

    # --- Reranking Logic ---
    reranked_uuids: list[str] = []
    search_result_uuids_by_method = [[node.uuid for node in result] for result in search_results_list]

    if config.reranker == NodeReranker.rrf or config.reranker == NodeReranker.node_distance or config.reranker == NodeReranker.episode_mentions:
        # Base ranking using RRF
        reranked_uuids_base = rrf(search_result_uuids_by_method)

        if config.reranker == NodeReranker.node_distance:
            if center_node_uuid is None:
                logger.warning('Node Distance reranker selected but no center_node_uuid provided. Falling back to RRF.')
                reranked_uuids = reranked_uuids_base
            else:
                reranked_uuids = await node_distance_reranker(driver, reranked_uuids_base, center_node_uuid)
        elif config.reranker == NodeReranker.episode_mentions:
            reranked_uuids = await episode_mentions_reranker(driver, search_result_uuids_by_method)
        else:
            # Simple RRF
            reranked_uuids = reranked_uuids_base

        # --- Post-RRF/NodeDistance Boosting ---
        # Avoid double boosting if episode_mentions already considers it implicitly
        if (priority_channels or boost_human) and config.reranker != NodeReranker.episode_mentions:
            logger.debug(f"Applying boost to Node RRF/NodeDistance results. Priority Channels: {priority_channels}, Boost Human: {boost_human}")

            # --- FIX START: Handle missing UUIDs and query episode props efficiently ---
            # Bulk query to get relevant episode properties for ranked nodes
            query_linked_ep_props: LiteralString = """
                UNWIND $node_uuids AS node_uuid
                MATCH (node:Entity {uuid: node_uuid})<-[:MENTIONS]-(ep:Episodic)
                RETURN node.uuid as node_uuid,
                       COLLECT(DISTINCT {uuid: ep.uuid, channel_name: ep.source_channel_name, is_human: ep.is_human_message}) as episodes
            """
            valid_reranked_uuids_for_boost = [uuid for uuid in reranked_uuids if uuid in node_uuid_map]
            if not valid_reranked_uuids_for_boost: # Avoid query if no valid nodes
                 node_episode_props_map = {}
            else:
                try:
                    linked_ep_records, _, _ = await driver.execute_query(
                        query_linked_ep_props, node_uuids=valid_reranked_uuids_for_boost, database_=DEFAULT_DATABASE, routing_='r'
                    )
                    node_episode_props_map: Dict[str, List[Dict]] = {rec['node_uuid']: rec['episodes'] for rec in linked_ep_records}
                except Exception as e:
                    logger.error(f"Failed to fetch linked episode properties for node boosting: {e}", exc_info=True)
                    node_episode_props_map = {}
            # --- FIX END ---


            boosted_scores: Dict[str, float] = {}
            for i, uuid in enumerate(reranked_uuids): # Iterate original ranked list
                base_score = 1 / (i + 1) # RRF-like score
                boost = 0.0
                # Check if this uuid is valid and has linked episodes
                if uuid in valid_reranked_uuids_for_boost:
                    linked_episodes = node_episode_props_map.get(uuid, [])
                    for ep_props in linked_episodes:
                        if ep_props:
                            is_priority = priority_channels and ep_props.get('channel_name') in priority_channels
                            is_human = boost_human and ep_props.get('is_human', True)
                            if is_priority or is_human:
                                boost = priority_boost
                                break # Apply boost once per node
                # Assign score even if boost is 0 or node was skipped
                boosted_scores[uuid] = base_score + boost

            reranked_uuids = sorted(boosted_scores, key=boosted_scores.get, reverse=True)
            logger.debug(f"Top 5 boosted node UUIDs: {reranked_uuids[:5]}")


    elif config.reranker == NodeReranker.mmr:
        candidates = [
            (node.uuid, node.name_embedding or [0.0] * 1024)
            for node in node_uuid_map.values() if node.name_embedding
        ]
        if candidates:
            reranked_uuids = maximal_marginal_relevance(
                query_vector, candidates, config.mmr_lambda
            )
        else:
            reranked_uuids = []

    elif config.reranker == NodeReranker.cross_encoder:
        rrf_base_uuids = rrf(search_result_uuids_by_method)
        candidates_for_ce = [node_uuid_map[uuid] for uuid in rrf_base_uuids if uuid in node_uuid_map][:limit * 2]

        summary_to_uuid_map = {node.summary: node.uuid for node in candidates_for_ce if node.summary}
        if summary_to_uuid_map:
            reranked_summaries = await cross_encoder.rank(query, list(summary_to_uuid_map.keys()))
            reranked_uuids = [summary_to_uuid_map[summary] for summary, _ in reranked_summaries]
        else:
            reranked_uuids = []
    else:
        logger.warning(f"Unknown node reranker '{config.reranker}', defaulting to RRF.")
        reranked_uuids = rrf(search_result_uuids_by_method)


    # --- Final node list construction ---
    final_nodes: List[EntityNode] = []
    seen_uuids = set()
    for uuid in reranked_uuids:
         # --- FIX: Check if uuid exists in map before accessing ---
         if uuid in node_uuid_map and uuid not in seen_uuids:
             final_nodes.append(node_uuid_map[uuid])
             seen_uuids.add(uuid)
         # --- FIX END ---

    logger.info(f"Node search returned {len(final_nodes[:limit])} results using {config.reranker}.")
    return final_nodes[:limit]


async def community_search(
    driver: AsyncDriver,
    cross_encoder: CrossEncoderClient,
    query: str,
    query_vector: list[float],
    group_ids: Optional[List[str]],
    config: Optional[CommunitySearchConfig],
    limit=DEFAULT_SEARCH_LIMIT,
     # Note: Priority boosting not implemented for communities yet
) -> list[CommunityNode]:
    if config is None:
        return []

    # --- Gather initial search results ---
    search_tasks = []
    if CommunitySearchMethod.bm25 in config.search_methods:
        search_tasks.append(community_fulltext_search(driver, query, group_ids, 2 * limit))
    if CommunitySearchMethod.cosine_similarity in config.search_methods:
         search_tasks.append(community_similarity_search(
            driver, query_vector, group_ids, 2 * limit, config.sim_min_score
        ))

    search_results_list: list[list[CommunityNode]] = list(await semaphore_gather(*search_tasks))

    # --- Deduplicate ---
    community_uuid_map: Dict[str, CommunityNode] = {comm.uuid: comm for result in search_results_list for comm in result}

    # --- Reranking ---
    reranked_uuids: list[str] = []
    search_result_uuids_by_method = [[comm.uuid for comm in result] for result in search_results_list]

    if config.reranker == CommunityReranker.rrf:
        reranked_uuids = rrf(search_result_uuids_by_method)

    elif config.reranker == CommunityReranker.mmr:
        candidates = [
            (comm.uuid, comm.name_embedding or [0.0] * 1024)
            for comm in community_uuid_map.values() if comm.name_embedding
        ]
        if candidates:
            reranked_uuids = maximal_marginal_relevance(
                query_vector, candidates, config.mmr_lambda
            )
        else:
            reranked_uuids = []

    elif config.reranker == CommunityReranker.cross_encoder:
        rrf_base_uuids = rrf(search_result_uuids_by_method)
        candidates_for_ce = [community_uuid_map[uuid] for uuid in rrf_base_uuids if uuid in community_uuid_map][:limit * 2]

        summary_to_uuid_map = {comm.summary: comm.uuid for comm in candidates_for_ce if comm.summary}
        if summary_to_uuid_map:
            reranked_summaries = await cross_encoder.rank(query, list(summary_to_uuid_map.keys()))
            reranked_uuids = [summary_to_uuid_map[summary] for summary, _ in reranked_summaries]
        else:
            reranked_uuids = []
    else:
        logger.warning(f"Unknown community reranker '{config.reranker}', defaulting to RRF.")
        reranked_uuids = rrf(search_result_uuids_by_method)

    # --- Final list ---
    final_communities: List[CommunityNode] = []
    seen_uuids = set()
    for uuid in reranked_uuids:
        # --- FIX: Check if uuid exists in map before accessing ---
        if uuid in community_uuid_map and uuid not in seen_uuids:
             final_communities.append(community_uuid_map[uuid])
             seen_uuids.add(uuid)
        # --- FIX END ---


    logger.info(f"Community search returned {len(final_communities[:limit])} results using {config.reranker}.")
    return final_communities[:limit]

