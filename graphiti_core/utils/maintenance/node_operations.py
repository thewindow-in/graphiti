# graphiti_core/utils/maintenance/node_operations.py
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
- Handle DYNAMIC/RECURRING EVENT NODES.
- Accept full EpisodeNode object for context.
- Set created_at for new EntityNodes based on Episode's valid_at time.
"""

import logging
from time import time
# --- MODIFICATION START: Add imports ---
from typing import List, Dict, Tuple, Optional, Set, Type
# --- MODIFICATION END ---

import pydantic
from pydantic import BaseModel

# --- MODIFICATION START: Import EntityEdge ---
from graphiti_core.edges import EntityEdge
# --- MODIFICATION END ---
from graphiti_core.helpers import MAX_REFLEXION_ITERATIONS, semaphore_gather
from graphiti_core.llm_client import LLMClient
# --- MODIFICATION START: Import EpisodicNode ---
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
# --- MODIFICATION END ---
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate
from graphiti_core.prompts.extract_nodes import (
    EntityClassification,
    ExtractedNodes,
    MissedEntities,
)
from graphiti_core.prompts.summarize_nodes import Summary
# --- MODIFICATION START: Import utc_now ---
# from graphiti_core.utils.datetime_utils import utc_now # Keep if needed elsewhere, but use episode time for created_at
# --- MODIFICATION END ---


logger = logging.getLogger(__name__)

# Define labels that represent dynamic, potentially recurring events
DYNAMIC_EVENT_LABELS: Set[str] = {
    "TechnicalIssue",
    "UserBlocker",
    "PayoutIssue",
    "PerformanceDip",
    "Outage", # Add more specific event types relevant to Wishlink
}
# Define the relationship type for linking recurring events
PREVIOUS_INSTANCE_RELATION_TYPE = "PREVIOUS_INSTANCE"


async def extract_message_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode, # Accept full episode
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(), # Use episode's valid_at
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_message(context), response_model=ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_text_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode, # Accept full episode
    previous_episodes: list[EpisodicNode],
    custom_prompt='',
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(), # Use episode's valid_at
        'previous_episodes': [ep.content for ep in previous_episodes],
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_text(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_json_nodes(
    llm_client: LLMClient, episode: EpisodicNode, custom_prompt='' # Accept full episode
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'episode_timestamp': episode.valid_at.isoformat(), # Use episode's valid_at
        'source_description': episode.source_description,
        'custom_prompt': custom_prompt,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.extract_json(context), ExtractedNodes
    )
    extracted_node_names = llm_response.get('extracted_node_names', [])
    return extracted_node_names


async def extract_nodes_reflexion(
    llm_client: LLMClient,
    episode: EpisodicNode, # Accept full episode
    previous_episodes: list[EpisodicNode],
    node_names: list[str],
) -> list[str]:
    # Prepare context for LLM
    context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': node_names,
    }

    llm_response = await llm_client.generate_response(
        prompt_library.extract_nodes.reflexion(context), MissedEntities
    )
    missed_entities = llm_response.get('missed_entities', [])

    return missed_entities


async def extract_nodes(
    llm_client: LLMClient,
    episode: EpisodicNode, # Accept full episode object
    previous_episodes: list[EpisodicNode],
    entity_types: dict[str, BaseModel] | None = None,
) -> list[EntityNode]:
    start = time()
    # --- MODIFICATION START: Dynamic prompt adjustment based on episode context ---
    custom_extraction_prompt_prefix = ""
    # Define priority channels (consider moving to config)
    priority_channels = {
         "#whats-new-product", "#product-x-business", "#pods-sourcing", "#announcement-x-support"
    }
    is_priority_channel = episode.source_channel_name in priority_channels # Assuming source_channel_name is populated
    if is_priority_channel:
         custom_extraction_prompt_prefix += " This message is from a high-priority channel, pay extra attention to extracting detailed entities and nuances."
    if not episode.is_human_message:
         # Optional: De-prioritize bot message extraction
         # custom_extraction_prompt_prefix += " This is a system/bot message, focus only on explicitly stated core entities."
         pass
    # --- MODIFICATION END ---

    extracted_node_names: list[str] = []
    # Pass the dynamic prefix along with any existing custom_prompt logic
    current_custom_prompt = custom_extraction_prompt_prefix # Start with dynamic part

    entities_missed = True
    reflexion_iterations = 0
    while entities_missed and reflexion_iterations < MAX_REFLEXION_ITERATIONS:
        if episode.source == EpisodeType.message:
            extracted_node_names = await extract_message_nodes(
                llm_client, episode, previous_episodes, current_custom_prompt
            )
        elif episode.source == EpisodeType.text:
            extracted_node_names = await extract_text_nodes(
                llm_client, episode, previous_episodes, current_custom_prompt
            )
        elif episode.source == EpisodeType.json:
            extracted_node_names = await extract_json_nodes(llm_client, episode, current_custom_prompt)

        reflexion_iterations += 1
        if reflexion_iterations < MAX_REFLEXION_ITERATIONS:
            # Pass full episode for reflexion context if needed
            missing_entities = await extract_nodes_reflexion(
                llm_client, episode, previous_episodes, extracted_node_names
            )
            entities_missed = len(missing_entities) != 0
            if entities_missed:
                missing_prompt = ' The following entities were missed in a previous extraction: '
                missing_prompt += ', '.join(missing_entities)
                # Append to existing custom prompt
                current_custom_prompt = custom_extraction_prompt_prefix + missing_prompt
            else:
                # Reset if no entities were missed in this iteration, keep prefix
                current_custom_prompt = custom_extraction_prompt_prefix
        else:
             entities_missed = False

    # --- Classification (no change needed here conceptually) ---
    node_classification_context = {
        'episode_content': episode.content,
        'previous_episodes': [ep.content for ep in previous_episodes],
        'extracted_entities': extracted_node_names,
        'entity_types': {
            type_name: values.model_json_schema().get('description', '')
            for type_name, values in entity_types.items()
        } if entity_types is not None else {},
    }
    node_classifications: dict[str, str | None] = {}
    if entity_types is not None and extracted_node_names:
        try:
            llm_response = await llm_client.generate_response(
                prompt_library.extract_nodes.classify_nodes(node_classification_context),
                response_model=EntityClassification,
            )
            response_data = llm_response if isinstance(llm_response, dict) else {}
            entity_classifications = response_data.get('entity_classifications', [])
            node_classifications.update(
                {
                    entity_classification.get('name'): entity_classification.get('entity_type')
                    for entity_classification in entity_classifications
                    if isinstance(entity_classification, dict) and entity_classification.get('name')
                }
            )
        except Exception as e:
            logger.error(f"Error during node classification: {e}", exc_info=True)

    end = time()
    logger.debug(f'Extracted new nodes: {extracted_node_names} in {(end - start) * 1000} ms')

    # --- Convert to EntityNode, setting created_at from episode ---
    new_nodes = []
    # --- MODIFICATION START: Use episode's valid_at for created_at ---
    episode_creation_time = episode.valid_at # This holds the original message timestamp
    # --- MODIFICATION END ---
    for name in extracted_node_names:
        entity_type = node_classifications.get(name)
        if entity_types is None or entity_type not in entity_types:
            entity_type = None

        labels = ['Entity']
        if entity_type and entity_type not in ('None', 'null'):
             labels.append(entity_type)
        labels = sorted(list(set(labels)))

        new_node = EntityNode(
            name=name,
            group_id=episode.group_id,
            labels=labels,
            summary='', # Summary generated later
            # --- MODIFICATION START: Set created_at ---
            created_at=episode_creation_time,
            # --- MODIFICATION END ---
        )
        new_nodes.append(new_node)
        logger.debug(f'Created new node: {new_node.name} (UUID: {new_node.uuid}, Labels: {labels}, CreatedAt: {episode_creation_time})')

    return new_nodes


async def resolve_extracted_node(
    llm_client: LLMClient,
    extracted_node: EntityNode,
    existing_nodes: list[EntityNode],
    episode: Optional[EpisodicNode] = None, # Accept full episode
    previous_episodes: Optional[list[EpisodicNode]] = None, # Accept previous episodes
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[EntityNode, dict[str, str], Optional[EntityEdge]]: # Return type unchanged
    start = time()
    previous_instance_edge: Optional[EntityEdge] = None
    uuid_map: dict[str, str] = {}

    # Use episode context if available
    episode_content = episode.content if episode else ''
    prev_episode_content = [ep.content for ep in previous_episodes] if previous_episodes else []
    episode_creation_time = episode.valid_at if episode else None # For setting created_at on new edges

    # Prepare context for LLM deduplication check
    existing_nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'labels': node.labels, 'attributes': node.attributes}
        for node in existing_nodes
    ]
    extracted_node_context = {
        'uuid': extracted_node.uuid,
        'name': extracted_node.name,
        'labels': extracted_node.labels,
        'summary': extracted_node.summary,
    }
    dedupe_context = {
        'existing_nodes': existing_nodes_context,
        'extracted_nodes': extracted_node_context,
        'episode_content': episode_content,
        'previous_episodes': prev_episode_content,
    }

    # Prepare context for summary and attribute generation
    summary_context = {
        'node_name': extracted_node.name,
        'node_summary': extracted_node.summary,
        'episode_content': episode_content,
        'previous_episodes': prev_episode_content,
        'attributes': [],
    }

    entity_type_classes: tuple[Type[BaseModel], ...] = tuple()
    if entity_types is not None:
         # Ensure labels attribute exists and is iterable
        node_labels = getattr(extracted_node, 'labels', [])
        if isinstance(node_labels, list):
            entity_type_classes = entity_type_classes + tuple(
                filter(
                    lambda x: x is not None,
                    [entity_types.get(label) for label in node_labels if label != 'Entity']
                )
            )

    for entity_type_class in entity_type_classes:
        if hasattr(entity_type_class, 'model_fields'):
             summary_context['attributes'].extend(list(entity_type_class.model_fields.keys())) # type: ignore

    base_classes_for_model = entity_type_classes + (Summary,)
    valid_base_classes = tuple(cls for cls in base_classes_for_model if cls is not None)

    # Handle potential error if valid_base_classes is empty
    if not valid_base_classes:
         # Fallback or raise error - let's default to Summary only
         logger.warning(f"No valid base classes found for node {extracted_node.name}, using Summary only.")
         valid_base_classes = (Summary,)


    entity_attributes_model: Type[BaseModel] = pydantic.create_model(
        f'Generated_{extracted_node.uuid[:8]}',
        __base__=valid_base_classes,
    )

    # Run LLM calls concurrently
    dedupe_response, node_attributes_response = await semaphore_gather(
        llm_client.generate_response(
            prompt_library.dedupe_nodes.node(dedupe_context), response_model=NodeDuplicate
        ),
        llm_client.generate_response(
            prompt_library.summarize_nodes.summarize_context(summary_context),
            response_model=entity_attributes_model,
        ),
    )

    # Update extracted node with generated summary and attributes
    attributes_data = node_attributes_response if isinstance(node_attributes_response, dict) else {}
    extracted_node.summary = attributes_data.get('summary', extracted_node.summary)
    attributes_to_update = {k: v for k, v in attributes_data.items() if k != 'summary'}
    extracted_node.attributes.update(attributes_to_update)

    # Process deduplication response
    dedupe_data = dedupe_response if isinstance(dedupe_response, dict) else {}
    is_duplicate: bool = dedupe_data.get('is_duplicate', False)
    duplicate_uuid: str | None = dedupe_data.get('uuid', None)
    resolved_name = dedupe_data.get('name', extracted_node.name)

    node_to_return = extracted_node
    node_to_return.name = resolved_name

    is_dynamic_event = any(label in DYNAMIC_EVENT_LABELS for label in getattr(extracted_node, 'labels', []))

    if is_duplicate and duplicate_uuid:
        existing_node: Optional[EntityNode] = None
        for n in existing_nodes:
            if n.uuid == duplicate_uuid:
                existing_node = n
                break

        if existing_node:
            if is_dynamic_event:
                logger.debug(f"Dynamic event recurrence: New '{node_to_return.name}' ({node_to_return.uuid}) vs existing '{existing_node.name}' ({existing_node.uuid}). Creating edge.")
                # --- MODIFICATION START: Set edge created_at ---
                if episode_creation_time: # Only create if we have context
                    previous_instance_edge = EntityEdge(
                        source_node_uuid=node_to_return.uuid,
                        target_node_uuid=existing_node.uuid,
                        name=PREVIOUS_INSTANCE_RELATION_TYPE,
                        fact=f"{node_to_return.name} is a recurrence of {existing_node.name}",
                        group_id=node_to_return.group_id,
                        created_at=episode_creation_time, # Use episode time
                        episodes=[episode.uuid] if episode else [],
                    )
                else:
                    logger.warning("Cannot create PREVIOUS_INSTANCE edge without episode context (timestamp).")
                # --- MODIFICATION END ---
            else:
                logger.debug(f"Static duplicate: Merging '{extracted_node.name}' ({extracted_node.uuid}) into '{existing_node.name}' ({existing_node.uuid}).")
                if not existing_node.summary or len(extracted_node.summary) > len(existing_node.summary):
                     existing_node.summary = extracted_node.summary
                existing_node.attributes.update(extracted_node.attributes)
                existing_node.name = resolved_name
                # --- IMPORTANT: Merge Labels? ---
                # Consider merging labels if the new node had more specific labels
                # existing_node.labels = sorted(list(set(existing_node.labels + extracted_node.labels)))
                node_to_return = existing_node
                uuid_map[extracted_node.uuid] = existing_node.uuid
        else:
             logger.warning(f"Dedupe flagged duplicate of {duplicate_uuid}, but node not found.")

    end = time()
    log_msg = f"Resolved node: Extracted='{extracted_node.name}' ({extracted_node.uuid}) -> Final='{node_to_return.name}' ({node_to_return.uuid})"
    # ... (rest of logging) ...
    logger.debug(log_msg)

    # Return final node, map, and optional new edge (with correct created_at)
    return node_to_return, uuid_map, previous_instance_edge


async def resolve_extracted_nodes(
    llm_client: LLMClient,
    extracted_nodes: list[EntityNode],
    existing_nodes_lists: list[list[EntityNode]],
    episode: Optional[EpisodicNode] = None, # Accept full episode
    previous_episodes: Optional[list[EpisodicNode]] = None, # Accept previous episodes
    entity_types: dict[str, BaseModel] | None = None,
) -> tuple[list[EntityNode], dict[str, str], list[EntityEdge]]: # Return type unchanged

    uuid_map: dict[str, str] = {}
    resolved_nodes: list[EntityNode] = []
    new_edges: list[EntityEdge] = []

    # Run resolution concurrently, passing episode context down
    results: list[tuple[EntityNode, dict[str, str], Optional[EntityEdge]]] = list(
        await semaphore_gather(
            *[
                resolve_extracted_node(
                    llm_client,
                    extracted_node,
                    existing_nodes,
                    episode, # Pass episode
                    previous_episodes, # Pass previous episodes
                    entity_types,
                )
                for extracted_node, existing_nodes in zip(extracted_nodes, existing_nodes_lists)
            ]
        )
    )

    # Process results
    for node, partial_uuid_map, potential_new_edge in results:
        uuid_map.update(partial_uuid_map)
        resolved_nodes.append(node)
        if potential_new_edge:
            new_edges.append(potential_new_edge)

    # Return resolved nodes, map, and new edges (with correct created_at)
    return resolved_nodes, uuid_map, new_edges


async def dedupe_node_list(
    llm_client: LLMClient,
    nodes: list[EntityNode],
) -> tuple[list[EntityNode], dict[str, str]]:
    # This function is less critical for the core add_episode flow focusing on incremental updates.
    # No changes needed here for timestamp or prioritization logic.
    start = time()
    node_map = {node.uuid: node for node in nodes}

    nodes_context = [
        {'uuid': node.uuid, 'name': node.name, 'summary': node.summary} for node in nodes
    ]
    context = {'nodes': nodes_context}

    llm_response = await llm_client.generate_response(
        prompt_library.dedupe_nodes.node_list(context)
    )

    response_data = llm_response if isinstance(llm_response, dict) else {}
    nodes_data = response_data.get('nodes', [])

    end = time()
    logger.debug(f'Deduplicated nodes list: {len(nodes_data)} groups in {(end - start) * 1000} ms')

    unique_nodes = []
    uuid_map: dict[str, str] = {}
    for node_data in nodes_data:
         if not isinstance(node_data, dict) or 'uuids' not in node_data or not node_data['uuids']:
             logger.warning(f"Skipping invalid node group data: {node_data}")
             continue

         first_uuid = node_data['uuids'][0]
         node_instance: Optional[EntityNode] = node_map.get(first_uuid)

         if node_instance is None:
            logger.warning(f'Node {first_uuid} from dedupe group not found in initial map')
            continue

         if 'summary' in node_data:
             node_instance.summary = node_data['summary']
         unique_nodes.append(node_instance)

         for uuid_to_map in node_data['uuids'][1:]:
            if uuid_to_map != first_uuid:
                 uuid_map[uuid_to_map] = first_uuid

    return unique_nodes, uuid_map