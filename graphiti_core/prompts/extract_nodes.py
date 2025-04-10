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

MODIFIED FOR WISHLINK CONTEXT
"""

import json
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class ExtractedNodes(BaseModel):
    extracted_node_names: list[str] = Field(..., description='Name of the extracted entity')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="Names of entities that weren't extracted")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='UUID of the entity')
    name: str = Field(description='Name of the entity')
    entity_type: str | None = Field(
        default=None, description='Type of the entity. Must be one of the provided types or None'
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='List of entities classification triples.'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Added Wishlink context and specific guidelines
    sys_prompt = """You are an AI assistant specialized in extracting entities from conversational messages within the context of Wishlink, a social commerce platform connecting Creators and Brands.

    Wishlink Context:
    - Creators (influencers) collaborate with Brands on Campaigns.
    - Creators share personalized 'wishlinks' (URLs) to recommend products.
    - Sales and commissions are tracked via these links.
    - Key entities include: Creators, Brands, Campaigns, Products, Commissions, Payouts, Employees, Teams, Projects.
    - Dynamic events like Technical Issues, User Blockers, Payout Issues also occur.

    Your primary task is to identify and extract the speaker and other significant entities (both static like people/projects and dynamic like issues/events) mentioned in the conversation."""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context["episode_content"]}
    </CURRENT MESSAGE>

    {context['custom_prompt']}

    Given the above conversation and Wishlink context, extract entity nodes from the CURRENT MESSAGE that are explicitly or implicitly mentioned:

    Guidelines:
    1.  ALWAYS extract the speaker/actor as the first node (part before the colon). Use full names if known.
    2.  Extract significant static entities relevant to Wishlink: Creators, Brands, Campaigns, Products, Employees, Teams, Projects. Use specific, full names (e.g., "Campaign Q2 Launch", "Creator Priya Sharma").
    3.  Extract significant dynamic events or issues: Technical Issues (e.g., "API Outage 2025-04-08"), User Blockers (e.g., "Creator Onboarding Blocked"), Payout Issues (e.g., "March Payout Discrepancy"). Be specific and include dates if mentioned for uniqueness.
    4.  DO NOT create nodes for relationships or actions (these are handled separately).
    5.  DO NOT create nodes for generic temporal information like dates, times or years unless part of a specific event name (e.g., "API Outage 2025-04-08").
    6.  DO NOT extract entities mentioned *only* in PREVIOUS MESSAGES; use them only for context.
    7.  Extract user preferences or specific feature requests as distinct nodes if significant.

    Example:
    CURRENT MESSAGE: "riya_sharma: Hey team, Brand X's campaign 'SummerGlow' is live! @prakash_jha can you check the commission tracking for wishlink.com/rs/xyz? Seems like the dashboard update from yesterday caused an issue."
    Desired Output: ["riya_sharma", "Brand X", "SummerGlow Campaign", "prakash_jha", "wishlink.com/rs/xyz", "Dashboard Update Issue 2025-04-07"] (Assuming yesterday was 2025-04-07 based on context/timestamp)
    """
    # <<< CHANGE END >>>
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Added Wishlink context (less critical for JSON but included for consistency)
    sys_prompt = """You are an AI assistant that extracts entity nodes from JSON data, considering the context of Wishlink, a social commerce platform.
    Your primary task is to identify and extract relevant entities from JSON files based on the source description."""

    user_prompt = f"""
    <SOURCE DESCRIPTION>:
    {context["source_description"]}
    </SOURCE DESCRIPTION>
    <JSON>
    {context["episode_content"]}
    </JSON>

    {context['custom_prompt']}

    Given the above source description and JSON, extract relevant entity nodes from the provided JSON, keeping Wishlink context (Creators, Brands, Campaigns, Products, Links, Commissions, etc.) in mind:

    Guidelines:
    1. Extract key identifiers or entities the JSON represents (e.g., a "campaign_id", "creator_name", "brand_id", "product_sku").
    2. Extract significant event names or issue identifiers if present.
    3. Do NOT extract simple properties that are likely attributes (handled by edge extraction) unless they represent a core entity.
    4. Do NOT extract any properties that contain dates unless it's part of a unique identifier or event name.
    """
    # <<< CHANGE END >>>
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Added Wishlink context and specific guidelines
    sys_prompt = """You are an AI assistant specialized in extracting entities from text documents within the context of Wishlink, a social commerce platform connecting Creators and Brands.

Wishlink Context:
- Creators (influencers) collaborate with Brands on Campaigns.
- Creators share personalized 'wishlinks' (URLs) to recommend products.
- Sales and commissions are tracked via these links.
- Key entities include: Creators, Brands, Campaigns, Products, Commissions, Payouts, Employees, Teams, Projects.
- Dynamic events like Technical Issues, User Blockers, Payout Issues also occur.

Your primary task is to identify and extract significant entities (both static like people/projects and dynamic like issues/events) mentioned in the text."""

    user_prompt = f"""
<TEXT>
{context["episode_content"]}
</TEXT>

{context['custom_prompt']}

Given the above text and Wishlink context, extract entity nodes that are explicitly or implicitly mentioned:

Guidelines:
1.  Extract significant static entities relevant to Wishlink: Creators, Brands, Campaigns, Products, Employees, Teams, Projects. Use specific, full names (e.g., "Campaign Q2 Launch", "Creator Priya Sharma").
2.  Extract significant dynamic events or issues: Technical Issues (e.g., "API Outage Report April 8th"), User Blockers (e.g., "Creator Onboarding Funnel Blockage"), Payout Issues (e.g., "Q1 Payout Calculation Error"). Be specific and include dates if mentioned for uniqueness.
3.  Avoid creating nodes for relationships or actions (these are handled separately).
4.  Avoid creating nodes for generic temporal information like dates, times or years unless part of a specific event name.
5.  Be as explicit as possible in your node names, using full names and avoiding abbreviations.
"""
    # <<< CHANGE END >>>
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    # No changes needed here for context, reflexion is about completeness check
    sys_prompt = """You are an AI assistant that determines which entities have not been extracted from the given context"""

    user_prompt = f"""
<PREVIOUS MESSAGES>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</PREVIOUS MESSAGES>
<CURRENT MESSAGE>
{context["episode_content"]}
</CURRENT MESSAGE>

<EXTRACTED ENTITIES>
{context["extracted_entities"]}
</EXTRACTED ENTITIES>

Given the above previous messages, current message, and list of extracted entities; determine if any significant entities (people, projects, brands, creators, campaigns, specific issues, events etc.) haven't been extracted from the CURRENT MESSAGE.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    # No changes needed here, classification depends on types provided at runtime
    sys_prompt = """You are an AI assistant that classifies entity nodes given the context from which they were extracted"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context["episode_content"]}
    </CURRENT MESSAGE>

    <EXTRACTED ENTITIES>
    {context['extracted_entities']}
    </EXTRACTED ENTITIES>

    <ENTITY TYPES>
    {context['entity_types']}
    </ENTITY TYPES>

    Given the above conversation, extracted entities, and provided entity types and their descriptions, classify the extracted entities.

    Guidelines:
    1. Each entity must have exactly one type.
    2. Only use the provided ENTITY TYPES as types, do not use additional types to classify entities.
    3. If none of the provided entity types accurately classify an extracted node, the type should be set to None.
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'classify_nodes': classify_nodes,
}
