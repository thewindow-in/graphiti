# graphiti_core/bulk_processor.py
"""
Handles bulk processing of episodes, aiming to replicate the quality
of the single add_episode flow (reflexion, resolution, temporal checks)
while improving performance through batching and concurrency.
"""

import asyncio
import logging
from time import time
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Set, Any
from uuid import uuid4
from pydantic import BaseModel, Field
from collections import defaultdict

# Graphiti Core Imports (adjust paths if needed)
from graphiti_core.graphiti import Graphiti # To access clients, driver
from graphiti_core.nodes import EntityNode, EpisodicNode, EpisodeType
from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.helpers import semaphore_gather, MAX_REFLEXION_ITERATIONS # Import reflexion limit
from graphiti_core.prompts import prompt_library
# --- MODIFICATION START: Import specific prompt models needed ---
from graphiti_core.prompts.extract_edges import MissingFacts as EdgeMissingFacts
from graphiti_core.prompts.extract_nodes import MissedEntities as NodeMissingEntities
# --- MODIFICATION END ---
from graphiti_core.search.search_utils import get_relevant_nodes, get_relevant_edges
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.utils.maintenance.graph_data_operations import retrieve_episodes, EPISODE_WINDOW_LEN
from graphiti_core.utils.maintenance.node_operations import (
    extract_message_nodes, extract_text_nodes, extract_json_nodes,
    # extract_nodes_reflexion, # Use prompt directly
    resolve_extracted_node,
    DYNAMIC_EVENT_LABELS, PREVIOUS_INSTANCE_RELATION_TYPE
)
# --- MODIFICATION START: Remove incorrect import ---
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges, # Keep this one
    # extract_edges_reflexion, # REMOVE THIS - Function doesn't exist
    resolve_extracted_edge, build_episodic_edges
)
# --- MODIFICATION END ---
from graphiti_core.utils.maintenance.temporal_operations import (
    extract_edge_dates, get_edge_contradictions
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk, resolve_edge_pointers

logger = logging.getLogger(__name__)

# --- Data Structures (Unchanged) ---
class RawEpisode(BaseModel):
    """Input structure for raw data before creating EpisodicNode."""
    uuid: Optional[str] = None
    name: str
    content: str
    source_description: str
    source: EpisodeType
    reference_time: datetime
    group_id: str
    source_channel_id: Optional[str] = None
    source_channel_name: Optional[str] = None
    is_human_message: bool = True

class BulkProcessingResult(BaseModel):
    """Result object summarizing a bulk processing run."""
    processed_episodes: int
    created_or_updated_nodes: int
    created_or_updated_edges: int
    created_episodic_edges: int
    duration_seconds: float

# --- Helper: Fetch Context (Unchanged) ---
async def _bulk_prepare_context(
    driver,
    episodes: List[EpisodicNode]
) -> Dict[str, List[EpisodicNode]]:
    """Fetches previous episodes context for a batch of episodes concurrently."""
    logger.debug(f"Fetching previous episode context for {len(episodes)} episodes.")
    tasks = [
        retrieve_episodes(driver, ep.valid_at, last_n=EPISODE_WINDOW_LEN, group_ids=[ep.group_id])
        for ep in episodes
    ]
    results = await semaphore_gather(*tasks)
    context_map = {ep.uuid: prev_eps for ep, prev_eps in zip(episodes, results)}
    logger.debug("Previous episode context fetched.")
    return context_map

# --- Helper: Node Extraction (Reflexion implemented) ---
async def _bulk_extract_nodes(
    llm_client: LLMClient,
    episodes: List[EpisodicNode],
    episode_context_map: Dict[str, List[EpisodicNode]],
    entity_types: Optional[Dict[str, BaseModel]] = None,
) -> Dict[str, List[EntityNode]]:
    """Extracts nodes for a batch of episodes, including reflexion."""
    extracted_nodes_map = defaultdict(list)

    async def _extract_single_episode_nodes(episode: EpisodicNode):
        previous_episodes = episode_context_map.get(episode.uuid, [])
        nodes_for_episode: List[EntityNode] = []
        extracted_node_names: List[str] = []
        custom_prompt = "" # Add priority logic here if needed

        # Reflexion Loop
        for iteration in range(MAX_REFLEXION_ITERATIONS):
            current_custom_prompt = custom_prompt
            if iteration > 0 and missing_entities:
                 missing_prompt = ' The following entities were missed: ' + ', '.join(missing_entities)
                 current_custom_prompt += missing_prompt

            # Choose extraction function based on source type
            new_node_names_this_pass: List[str] = []
            if episode.source == EpisodeType.message:
                new_node_names_this_pass = await extract_message_nodes(llm_client, episode, previous_episodes, current_custom_prompt)
            elif episode.source == EpisodeType.text:
                new_node_names_this_pass = await extract_text_nodes(llm_client, episode, previous_episodes, current_custom_prompt)
            elif episode.source == EpisodeType.json:
                 new_node_names_this_pass = await extract_json_nodes(llm_client, episode, current_custom_prompt)

            # Add newly found names (avoid duplicates within the episode for now)
            newly_found_unique_names = [name for name in new_node_names_this_pass if name not in extracted_node_names]
            extracted_node_names.extend(newly_found_unique_names)

            # Reflexion check using the prompt
            if iteration < MAX_REFLEXION_ITERATIONS - 1:
                reflexion_context = {
                    'episode_content': episode.content,
                    'previous_episodes': [ep.content for ep in previous_episodes],
                    'extracted_entities': extracted_node_names, # Use all found so far
                }
                # --- Call Node Reflexion Prompt ---
                try:
                    reflexion_response = await llm_client.generate_response(
                        prompt_library.extract_nodes.reflexion(reflexion_context),
                        response_model=NodeMissingEntities
                    )
                    missing_entities = reflexion_response.get('missed_entities', [])
                    if not missing_entities:
                        logger.debug(f"Node reflexion complete for ep {episode.uuid}, no missing entities found.")
                        break # No missing entities found, exit loop
                    else:
                         logger.debug(f"Node reflexion found missing entities for ep {episode.uuid}: {missing_entities}")
                except Exception as e:
                     logger.error(f"Error during node reflexion call for ep {episode.uuid}: {e}", exc_info=True)
                     break # Stop reflexion on error
            else:
                logger.debug(f"Max node reflexion iterations reached for ep {episode.uuid}.")
                break # Exit after max iterations

        # Convert final list of names to EntityNode objects
        # TODO: Add classification logic here if needed using entity_types
        for name in extracted_node_names:
             labels = ['Entity']
             node = EntityNode(
                 name=name,
                 group_id=episode.group_id,
                 labels=labels,
                 created_at=episode.valid_at # Use original message time
             )
             nodes_for_episode.append(node)

        extracted_nodes_map[episode.uuid] = nodes_for_episode
        logger.debug(f"Extracted {len(nodes_for_episode)} nodes for episode {episode.uuid}")

    # Run extraction for all episodes concurrently
    await semaphore_gather(*[_extract_single_episode_nodes(ep) for ep in episodes])
    return extracted_nodes_map


# --- Helper: Node Resolution (Unchanged from previous version - needs careful testing) ---
async def _bulk_resolve_nodes(
    llm_client: LLMClient,
    driver, # Neo4j driver
    embedder: EmbedderClient,
    extracted_nodes_by_episode: Dict[str, List[EntityNode]],
    episodes_map: Dict[str, EpisodicNode], # Map uuid to episode obj
    episode_context_map: Dict[str, List[EpisodicNode]],
    entity_types: Optional[Dict[str, BaseModel]] = None,
) -> Tuple[Dict[str, EntityNode], Dict[str, str], List[EntityEdge], Dict[str, List[str]]]:
    """
    Resolves nodes in bulk: generates embeddings, fetches relevant existing nodes,
    runs deduplication/summarization concurrently, manages UUID mapping.
    Returns:
        - all_resolved_nodes_map: {final_uuid: EntityNode}
        - final_uuid_map: {original_extracted_uuid: final_uuid}
        - structural_edges: List of new edges like PREVIOUS_INSTANCE
        - episode_to_resolved_node_uuids: {episode_uuid: [final_node_uuids]}
    """
    final_uuid_map = {} # {original_uuid: final_uuid}
    structural_edges = []
    all_resolved_nodes_map = {} # {final_uuid: EntityNode} - Stores the definitive version
    episode_to_resolved_node_uuids = defaultdict(list) # {ep_uuid: [final_uuids]}

    # 1. Flatten list and generate embeddings concurrently
    all_extracted_nodes: List[EntityNode] = []
    nodes_by_uuid: Dict[str, EntityNode] = {}
    for ep_uuid, nodes in extracted_nodes_by_episode.items():
        for node in nodes:
            if node.uuid in nodes_by_uuid: logger.warning(f"Duplicate extracted node UUID {node.uuid} found.")
            nodes_by_uuid[node.uuid] = node
            all_extracted_nodes.append(node)

    logger.debug(f"Generating embeddings for {len(all_extracted_nodes)} extracted nodes...")
    await semaphore_gather(*[
        node.generate_name_embedding(embedder)
        for node in all_extracted_nodes if node.name_embedding is None
    ])
    logger.debug("Node embeddings generated.")

    # 2. Fetch relevant existing nodes (Optimize this - using per-node for now)
    logger.debug("Fetching relevant existing nodes for resolution...")
    relevant_nodes_tasks = [
        get_relevant_nodes(driver, SearchFilters(), [node])
        for node in all_extracted_nodes
    ]
    relevant_nodes_results = await semaphore_gather(*relevant_nodes_tasks)
    relevant_nodes_map: Dict[str, List[EntityNode]] = {
        node.uuid: relevant_nodes_results[i]
        for i, node in enumerate(all_extracted_nodes)
    }
    logger.debug("Relevant existing nodes fetched.")

    # 3. Resolve each node concurrently
    logger.debug(f"Resolving {len(all_extracted_nodes)} nodes...")
    resolve_tasks = []
    node_to_episode_context: Dict[str, Tuple[EpisodicNode, List[EpisodicNode]]] = {}
    for ep_uuid, nodes in extracted_nodes_by_episode.items():
         episode = episodes_map[ep_uuid]
         prev_eps = episode_context_map.get(ep_uuid, [])
         for node in nodes:
             node_to_episode_context[node.uuid] = (episode, prev_eps)

    for node in all_extracted_nodes:
        existing_context = relevant_nodes_map.get(node.uuid, [])
        episode_context = node_to_episode_context.get(node.uuid)
        if episode_context:
            episode, prev_eps = episode_context
            # Pass episode time for potential PREVIOUS_INSTANCE edge creation
            task = resolve_extracted_node(
                llm_client, node, existing_context, episode, prev_eps, entity_types
            )
            resolve_tasks.append(task)
        else:
             logger.warning(f"Could not find episode context for extracted node {node.uuid}. Skipping resolution.")

    resolution_results: List[Tuple[EntityNode, Dict[str, str], Optional[EntityEdge]]] = await semaphore_gather(*resolve_tasks)
    logger.debug("Node resolution LLM calls complete.")

    # 4. Consolidate results (Sequentially to manage maps safely)
    logger.debug("Consolidating node resolution results...")
    temp_uuid_map = {}
    for original_node, node_uuid_map_part, potential_edge in resolution_results:
        final_node = original_node
        final_uuid = final_node.uuid
        original_extracted_uuid = list(node_uuid_map_part.keys())[0] if node_uuid_map_part else final_uuid
        temp_uuid_map[original_extracted_uuid] = final_uuid
        all_resolved_nodes_map[final_uuid] = final_node
        if potential_edge:
            # Ensure structural edge has correct created_at time (should be set by resolve_extracted_node)
            if potential_edge.created_at is None:
                 logger.warning(f"Structural edge {potential_edge.uuid} missing created_at. Defaulting.")
                 # Find episode time or use now()
                 ep_context = node_to_episode_context.get(original_extracted_uuid)
                 potential_edge.created_at = ep_context[0].valid_at if ep_context else datetime.now(timezone.utc)
            structural_edges.append(potential_edge)

    # Resolve chained mappings
    final_uuid_map = {}
    processed_in_chain = set()
    def get_final_uuid(start_uuid):
        # ... (chain resolution logic - same as before) ...
        path = [start_uuid]
        current = start_uuid
        while current in temp_uuid_map and temp_uuid_map[current] != current:
            next_uuid = temp_uuid_map[current]
            if next_uuid in path: logger.error(f"Cycle detected in UUID map involving {next_uuid}. Breaking."); return current
            path.append(next_uuid)
            current = next_uuid
            processed_in_chain.add(current)
        return current

    for original_uuid in temp_uuid_map:
         if original_uuid not in processed_in_chain:
             final_target_uuid = get_final_uuid(original_uuid)
             current_map_uuid = original_uuid
             while current_map_uuid in temp_uuid_map and current_map_uuid not in final_uuid_map:
                  final_uuid_map[current_map_uuid] = final_target_uuid
                  next_map_uuid = temp_uuid_map[current_map_uuid]
                  if next_map_uuid == current_map_uuid: break
                  current_map_uuid = next_map_uuid

    # Ensure all original nodes map to something
    for node in all_extracted_nodes:
        if node.uuid not in final_uuid_map: final_uuid_map[node.uuid] = node.uuid
        final_target = final_uuid_map[node.uuid]
        if final_target not in all_resolved_nodes_map:
             logger.warning(f"Node {node.uuid} resolved to {final_target}, which is not in the final map. Using original node.")
             all_resolved_nodes_map[node.uuid] = nodes_by_uuid[node.uuid]
             final_uuid_map[node.uuid] = node.uuid

    # 5. Map episodes to their final resolved node UUIDs
    for ep_uuid, extracted_ep_nodes in extracted_nodes_by_episode.items():
        resolved_uuids_for_ep = {final_uuid_map.get(n.uuid, n.uuid) for n in extracted_ep_nodes}
        episode_to_resolved_node_uuids[ep_uuid] = sorted(list(resolved_uuids_for_ep))

    logger.debug("Node consolidation complete.")
    return all_resolved_nodes_map, final_uuid_map, structural_edges, episode_to_resolved_node_uuids


# --- Helper: Edge Extraction (Implementing Reflexion) ---
async def _bulk_extract_edges(
    llm_client: LLMClient,
    episodes_map: Dict[str, EpisodicNode],
    episode_context_map: Dict[str, List[EpisodicNode]],
    resolved_nodes_by_episode: Dict[str, List[EntityNode]],
) -> Dict[str, List[EntityEdge]]:
    """Extracts edges for a batch of episodes using resolved nodes, including reflexion."""
    extracted_edges_map = defaultdict(list)

    async def _extract_single_episode_edges(episode_uuid: str):
        episode = episodes_map[episode_uuid]
        previous_episodes = episode_context_map.get(episode.uuid, [])
        resolved_nodes_for_context = resolved_nodes_by_episode.get(episode.uuid, [])
        if not resolved_nodes_for_context: # Skip if no nodes for context
             extracted_edges_map[episode_uuid] = []
             return

        edges_for_episode: List[EntityEdge] = []
        extracted_facts_this_episode: List[str] = []
        custom_prompt = "" # Add priority logic if needed

        # Reflexion Loop for Edges
        for iteration in range(MAX_REFLEXION_ITERATIONS):
            current_custom_prompt = custom_prompt
            if iteration > 0 and missing_facts:
                 missing_prompt = ' The following facts were missed: ' + ', '.join(missing_facts)
                 current_custom_prompt += missing_prompt

            # Call single extract_edges - assumes it uses episode time for created_at
            # NOTE: Ensure extract_edges uses episode.valid_at for new edge created_at.
            # Pass custom prompt if supported.
            current_edges_this_pass = await extract_edges(
                llm_client, episode, resolved_nodes_for_context, previous_episodes, episode.group_id
                # custom_prompt=current_custom_prompt # Uncomment if extract_edges accepts it
            )

            # Add newly found edges (avoid duplicates based on simple fact string for now)
            newly_found_edges = []
            for edge in current_edges_this_pass:
                 if edge.fact not in extracted_facts_this_episode:
                      edges_for_episode.append(edge)
                      extracted_facts_this_episode.append(edge.fact)
                      newly_found_edges.append(edge)

            logger.debug(f"Edge extraction pass {iteration+1} for ep {episode.uuid} found {len(newly_found_edges)} new edges.")

            # Reflexion check using the prompt
            if iteration < MAX_REFLEXION_ITERATIONS - 1:
                reflexion_context = {
                    'episode_content': episode.content,
                    'previous_episodes': [ep.content for ep in previous_episodes],
                    'nodes': [n.name for n in resolved_nodes_for_context], # Pass node names
                    'extracted_facts': extracted_facts_this_episode # Use all facts found so far
                }
                # --- Call Edge Reflexion Prompt ---
                try:
                    reflexion_response = await llm_client.generate_response(
                        prompt_library.extract_edges.reflexion(reflexion_context),
                        response_model=EdgeMissingFacts
                    )
                    missing_facts = reflexion_response.get('missing_facts', [])
                    if not missing_facts:
                        logger.debug(f"Edge reflexion complete for ep {episode.uuid}, no missing facts found.")
                        break # No missing facts found, exit loop
                    else:
                         logger.debug(f"Edge reflexion found missing facts for ep {episode.uuid}: {missing_facts}")
                except Exception as e:
                     logger.error(f"Error during edge reflexion call for ep {episode.uuid}: {e}", exc_info=True)
                     break # Stop reflexion on error
            else:
                logger.debug(f"Max edge reflexion iterations reached for ep {episode.uuid}.")
                break # Exit after max iterations

        extracted_edges_map[episode_uuid] = edges_for_episode
        logger.debug(f"Extracted {len(edges_for_episode)} final edges for episode {episode.uuid}")

    # Run extraction for all episodes concurrently
    await semaphore_gather(*[_extract_single_episode_edges(ep_uuid) for ep_uuid in episodes_map.keys()])
    return extracted_edges_map


# --- Helper: Edge Resolution (Unchanged from previous version - needs careful testing) ---
async def _bulk_resolve_edges(
    llm_client: LLMClient,
    driver, # Neo4j driver
    embedder: EmbedderClient,
    extracted_edges_by_episode: Dict[str, List[EntityEdge]],
    episodes_map: Dict[str, EpisodicNode],
    episode_context_map: Dict[str, List[EpisodicNode]],
    final_uuid_map: Dict[str, str], # Map from original extracted node UUIDs to final ones
) -> Tuple[List[EntityEdge], List[EntityEdge]]:
    """
    Resolves edges in bulk: resolves pointers, generates embeddings, fetches relevant,
    runs dedupe/temporal/contradiction checks concurrently.
    Returns:
        - final_resolved_edges: List of edges to be saved (new or existing updated)
        - final_invalidated_edges: List of existing edges marked as invalidated
    """
    final_resolved_edges_map: Dict[str, EntityEdge] = {} # {edge_uuid: Edge}
    final_invalidated_edges_map: Dict[str, EntityEdge] = {} # {edge_uuid: Edge}

    # 1. Flatten edge list & Resolve pointers
    all_extracted_edges: List[EntityEdge] = []
    edge_to_episode_uuid: Dict[str, str] = {} # Track origin episode for context
    for ep_uuid, edges in extracted_edges_by_episode.items():
        for edge in edges:
             all_extracted_edges.append(edge)
             edge_to_episode_uuid[edge.uuid] = ep_uuid # Store mapping

    resolve_edge_pointers(all_extracted_edges, final_uuid_map)
    logger.debug(f"Resolved pointers for {len(all_extracted_edges)} extracted edges.")

    # 2. Generate embeddings concurrently
    logger.debug(f"Generating embeddings for {len(all_extracted_edges)} extracted edges...")
    await semaphore_gather(*[
        edge.generate_embedding(embedder)
        for edge in all_extracted_edges if edge.fact_embedding is None
    ])
    logger.debug("Edge embeddings generated.")

    # 3. Fetch relevant existing edges (Optimize this)
    logger.debug("Fetching relevant existing edges for resolution...")
    relevant_edges_tasks = []
    edge_to_fetch_context_map = {}
    processed_pairs = set()

    for edge in all_extracted_edges:
         pair_key = tuple(sorted((edge.source_node_uuid, edge.target_node_uuid)))
         if pair_key not in processed_pairs:
             relevant_edges_tasks.append(
                 get_relevant_edges(driver, [edge], edge.source_node_uuid, edge.target_node_uuid)
             )
             processed_pairs.add(pair_key)
         # Store context needed for this edge's resolution later
         ep_uuid = edge_to_episode_uuid.get(edge.uuid)
         if ep_uuid and ep_uuid in episodes_map:
              edge_to_fetch_context_map[edge.uuid] = {
                   "episode": episodes_map[ep_uuid],
                   "previous_episodes": episode_context_map.get(ep_uuid, []),
                   "pair_key": pair_key
              }
         else:
              logger.warning(f"Could not find original episode context for edge {edge.uuid}")

    relevant_edges_results_list = await semaphore_gather(*relevant_edges_tasks)
    relevant_edges_by_pair: Dict[Tuple[str, str], List[EntityEdge]] = {}
    pair_list = list(processed_pairs)
    for i, results in enumerate(relevant_edges_results_list):
         pair_key = pair_list[i]
         relevant_edges_by_pair[pair_key] = results
    logger.debug("Relevant existing edges fetched.")

    # 4. Fetch existing edges for contradiction checks (Optimize this)
    all_involved_nodes = {edge.source_node_uuid for edge in all_extracted_edges} | \
                         {edge.target_node_uuid for edge in all_extracted_edges}
    logger.debug(f"Fetching contradiction context edges for {len(all_involved_nodes)} nodes...")
    contradiction_context_edges_tasks = [
        EntityEdge.get_by_node_uuid(driver, node_uuid) for node_uuid in all_involved_nodes
    ]
    contradiction_edges_results = await semaphore_gather(*contradiction_context_edges_tasks)
    contradiction_context_map: Dict[str, List[EntityEdge]] = {}
    for node_uuid, edges in zip(all_involved_nodes, contradiction_edges_results):
         contradiction_context_map[node_uuid] = edges
    logger.debug("Contradiction context edges fetched.")

    # 5. Resolve each edge concurrently
    logger.debug(f"Resolving {len(all_extracted_edges)} edges...")
    resolve_tasks = []
    for edge in all_extracted_edges:
        fetch_context = edge_to_fetch_context_map.get(edge.uuid)
        if not fetch_context:
             logger.warning(f"Skipping resolution for edge {edge.uuid} due to missing context.")
             continue

        episode = fetch_context["episode"]
        previous_episodes = fetch_context["previous_episodes"]
        pair_key = fetch_context["pair_key"]
        related_edges_for_dedupe = relevant_edges_by_pair.get(pair_key, [])
        source_contradiction_ctx = contradiction_context_map.get(edge.source_node_uuid, [])
        target_contradiction_ctx = contradiction_context_map.get(edge.target_node_uuid, [])
        contradiction_ctx_map = {e.uuid: e for e in source_contradiction_ctx + target_contradiction_ctx}
        existing_edges_for_contradiction = list(contradiction_ctx_map.values())

        # NOTE: Ensure resolve_extracted_edge uses episode time for created_at if edge is new
        task = resolve_extracted_edge(
            llm_client, edge, related_edges_for_dedupe, existing_edges_for_contradiction,
            episode, previous_episodes,
        )
        resolve_tasks.append(task)

    resolution_results: List[Tuple[EntityEdge, List[EntityEdge]]] = await semaphore_gather(*resolve_tasks)
    logger.debug("Edge resolution LLM calls complete.")

    # 6. Consolidate results
    logger.debug("Consolidating edge resolution results...")
    for resolved_edge, invalidated_edge_chunk in resolution_results:
        final_resolved_edges_map[resolved_edge.uuid] = resolved_edge
        for inv_edge in invalidated_edge_chunk:
            final_invalidated_edges_map[inv_edge.uuid] = inv_edge

    logger.debug("Edge consolidation complete.")
    return list(final_resolved_edges_map.values()), list(final_invalidated_edges_map.values())


# --- Main Bulk Processing Function ---
async def process_episodes_bulk(
    graphiti: Graphiti, # Pass Graphiti instance for clients/driver
    raw_episodes_batch: List[RawEpisode], # Operate on one batch of raw data
    entity_types: Optional[Dict[str, BaseModel]] = None,
    store_raw_content: bool = True, # Flag from Graphiti instance
) -> BulkProcessingResult:
    """
    Processes a batch of raw episode data using the refined bulk logic.
    """
    start_time = time()
    logger.info(f"Starting bulk processing for batch of {len(raw_episodes_batch)} episodes.")

    # 1. Create EpisodicNode objects
    episodic_nodes: List[EpisodicNode] = []
    episodes_map: Dict[str, EpisodicNode] = {}
    for raw_episode in raw_episodes_batch:
        ep_uuid = raw_episode.uuid or str(uuid4()) # Use provided or generate UUID
        # Ensure reference_time is timezone-aware (UTC)
        ref_time = raw_episode.reference_time
        if ref_time.tzinfo is None:
             ref_time = ref_time.replace(tzinfo=timezone.utc)
        else:
             ref_time = ref_time.astimezone(timezone.utc)

        episode = EpisodicNode(
            uuid=ep_uuid,
            name=raw_episode.name,
            content=raw_episode.content,
            source_description=raw_episode.source_description,
            source=raw_episode.source,
            created_at=ref_time, # Use original time
            valid_at=ref_time,
            group_id=raw_episode.group_id,
            source_channel_id=raw_episode.source_channel_id,
            source_channel_name=raw_episode.source_channel_name,
            is_human_message=raw_episode.is_human_message,
        )
        episodic_nodes.append(episode)
        episodes_map[ep_uuid] = episode

    # 2. Prepare Context
    episode_context_map = await _bulk_prepare_context(graphiti.driver, episodic_nodes)

    # 3. Extract Nodes
    extracted_nodes_by_episode = await _bulk_extract_nodes(
        graphiti.llm_client, episodic_nodes, episode_context_map, entity_types
    )

    # 4. Resolve Nodes
    resolved_nodes_map, final_uuid_map, structural_edges, episode_to_resolved_node_uuids = await _bulk_resolve_nodes(
        graphiti.llm_client, graphiti.driver, graphiti.embedder,
        extracted_nodes_by_episode, episodes_map, episode_context_map, entity_types
    )

    # 5. Extract Edges
    resolved_nodes_for_edge_extraction: Dict[str, List[EntityNode]] = defaultdict(list)
    for ep_uuid, resolved_node_uuids in episode_to_resolved_node_uuids.items():
        for node_uuid in resolved_node_uuids:
            if node_uuid in resolved_nodes_map:
                 resolved_nodes_for_edge_extraction[ep_uuid].append(resolved_nodes_map[node_uuid])

    extracted_edges_by_episode = await _bulk_extract_edges(
        graphiti.llm_client, episodes_map, episode_context_map, resolved_nodes_for_edge_extraction
    )

    # 6. Resolve Edges
    final_resolved_edges, final_invalidated_edges = await _bulk_resolve_edges(
        graphiti.llm_client, graphiti.driver, graphiti.embedder,
        extracted_edges_by_episode, episodes_map, episode_context_map, final_uuid_map
    )

    # 7. Prepare Final Data & Save
    logger.info("Preparing final data for saving...")
    all_final_entity_edges_map = {edge.uuid: edge for edge in structural_edges}
    all_final_entity_edges_map.update({edge.uuid: edge for edge in final_resolved_edges})
    all_final_entity_edges_map.update({edge.uuid: edge for edge in final_invalidated_edges})

    processed_episode_nodes = []
    all_episodic_edges = []
    edges_by_episode_uuid = defaultdict(set)

    # Link episodes to final entity edges
    for edge in all_final_entity_edges_map.values():
         if edge.created_at is None:
              logger.warning(f"Edge {edge.uuid} missing created_at before save. Defaulting.")
              fallback_time = datetime.now(timezone.utc)
              if edge.episodes and episodes_map.get(edge.episodes[0]): fallback_time = episodes_map[edge.episodes[0]].valid_at
              edge.created_at = fallback_time
         for ep_uuid in edge.episodes: edges_by_episode_uuid[ep_uuid].add(edge.uuid)

    # Prepare final episode nodes and create episodic edges
    for episode in episodic_nodes:
        episode.entity_edges = sorted(list(edges_by_episode_uuid.get(episode.uuid, set())))
        if not store_raw_content: episode.content = ''
        processed_episode_nodes.append(episode)

        # Create Episodic Edges
        resolved_node_uuids_for_ep = episode_to_resolved_node_uuids.get(episode.uuid, [])
        nodes_for_ep = [resolved_nodes_map[uuid] for uuid in resolved_node_uuids_for_ep if uuid in resolved_nodes_map]
        # Ensure nodes_for_ep contains EntityNode objects
        valid_nodes_for_ep = [n for n in nodes_for_ep if isinstance(n, EntityNode)]
        if len(valid_nodes_for_ep) != len(nodes_for_ep):
             logger.warning(f"Mismatch creating episodic edges for {episode.uuid}. Some resolved nodes might be missing or of wrong type.")
        all_episodic_edges.extend(build_episodic_edges(valid_nodes_for_ep, episode, episode.valid_at)) # Use original time

    final_entity_nodes_to_save = list(resolved_nodes_map.values())
    for node in final_entity_nodes_to_save:
         if node.created_at is None:
             logger.warning(f"Node {node.uuid} missing created_at before save. Defaulting.")
             # Try to find an episode time, else use now()
             fallback_time = datetime.now(timezone.utc)
             for ep_uuid, node_uuids in episode_to_resolved_node_uuids.items():
                  if node.uuid in node_uuids and ep_uuid in episodes_map:
                       fallback_time = episodes_map[ep_uuid].valid_at
                       break
             node.created_at = fallback_time


    final_entity_edges_to_save = list(all_final_entity_edges_map.values())

    logger.info(f"Final counts: Episodes={len(processed_episode_nodes)}, Nodes={len(final_entity_nodes_to_save)}, EntityEdges={len(final_entity_edges_to_save)}, EpisodicEdges={len(all_episodic_edges)}")

    # Bulk Save
    if processed_episode_nodes or final_entity_nodes_to_save or final_entity_edges_to_save or all_episodic_edges:
        await add_nodes_and_edges_bulk(
            graphiti.driver,
            processed_episode_nodes,
            all_episodic_edges,
            final_entity_nodes_to_save,
            final_entity_edges_to_save,
        )
    else:
         logger.info("No new data to save for this batch.")


    end_time = time()
    duration = end_time - start_time
    logger.info(f"Finished bulk processing batch of {len(raw_episodes_batch)} episodes in {duration:.2f} seconds.")

    return BulkProcessingResult(
        processed_episodes=len(processed_episode_nodes),
        created_or_updated_nodes=len(final_entity_nodes_to_save),
        created_or_updated_edges=len(final_entity_edges_to_save),
        created_episodic_edges=len(all_episodic_edges),
        duration_seconds=duration
    )

