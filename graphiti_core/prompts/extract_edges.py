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


class Edge(BaseModel):
    relation_type: str = Field(..., description='RELATION_TYPE_IN_CAPS (e.g., COLLABORATES_WITH, SHARED_LINK, TRACKS_COMMISSION, REPORTS_TO, ASSIGNED_TO, EXPERIENCES_ISSUE)')
    source_entity_name: str = Field(..., description='Name of the source entity (must be from the ENTITIES list)')
    target_entity_name: str = Field(..., description='Name of the target entity (must be from the ENTITIES list)')
    fact: str = Field(..., description='Extracted factual sentence describing the relationship between the source and target entities.')


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="Facts describing relationships between provided entities that weren't extracted")


class Prompt(Protocol):
    edge: PromptVersion
    reflexion: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Added Wishlink context and specific guidelines
    sys_prompt = """You are an expert fact extractor specialized in identifying relationships within the context of Wishlink, a social commerce platform connecting Creators and Brands.

    Wishlink Context:
    - Creators collaborate with Brands on Campaigns.
    - Creators share 'wishlinks' (URLs) to recommend products.
    - Sales and commissions are tracked.
    - Relationships involve: Creators, Brands, Campaigns, Products, Wishlinks, Commissions, Payouts, Employees, Teams, Projects, Technical Issues, User Blockers, etc.

    Your task is to extract factual relationships (triples) between the provided ENTITIES based on the CURRENT MESSAGE."""

    user_prompt = f"""
        <PREVIOUS MESSAGES>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </PREVIOUS MESSAGES>
        <CURRENT MESSAGE>
        {context["episode_content"]}
        </CURRENT MESSAGE>

        <ENTITIES>
        {context["nodes"]}
        </ENTITIES>

        {context['custom_prompt']}

        Given the above MESSAGES and the list of identified ENTITIES, extract all facts describing relationships *between* these ENTITIES from the CURRENT MESSAGE.

        Guidelines:
        1.  Extract facts ONLY between entities listed in the <ENTITIES> section.
        2.  Each fact must represent a clear relationship between two DISTINCT entities from the list.
        3.  The `relation_type` should be a concise, ALL_CAPS verb phrase describing the relationship (e.g., COLLABORATES_WITH, SHARED_LINK, TRACKS_COMMISSION, REPORTS_TO, ASSIGNED_TO, EXPERIENCES_ISSUE, IS_PART_OF, GENERATED_SALES).
        4.  The `fact` field should be a complete sentence extracted or synthesized from the message, clearly stating the relationship between the source and target entities. Include relevant details mentioned in the message.
        5.  Prioritize relationships relevant to Wishlink's operations:
            - Creator <-> Brand (COLLABORATES_WITH, PAID_BY)
            - Creator <-> Campaign (PARTICIPATES_IN)
            - Creator <-> Wishlink (SHARED_LINK, EARNS_COMMISSION_FROM)
            - Brand <-> Campaign (SPONSORS, RUNS)
            - Campaign <-> Product (FEATURES)
            - Employee <-> Project/Team (ASSIGNED_TO, WORKS_ON, MANAGES)
            - Employee/Team <-> TechnicalIssue/UserBlocker (REPORTS_ISSUE, EXPERIENCES_ISSUE, RESOLVES_ISSUE)
        6.  Consider temporal aspects if mentioned directly in relation to the fact (e.g., "assigned yesterday" - but the date extraction itself is handled later).
        7.  Do not extract facts only mentioned in PREVIOUS MESSAGES.

        Example:
        ENTITIES: ["riya_sharma", "Brand X", "SummerGlow Campaign", "prakash_jha", "wishlink.com/rs/xyz", "Dashboard Update Issue 2025-04-07"]
        CURRENT MESSAGE: "riya_sharma: Hey team, Brand X's campaign 'SummerGlow' is live! @prakash_jha can you check the commission tracking for wishlink.com/rs/xyz? Seems like the dashboard update from yesterday caused an issue."
        Desired Edges:
        [
          {{ "relation_type": "SPONSORS", "source_entity_name": "Brand X", "target_entity_name": "SummerGlow Campaign", "fact": "Brand X's campaign 'SummerGlow' is live." }},
          {{ "relation_type": "SHARED_LINK", "source_entity_name": "riya_sharma", "target_entity_name": "wishlink.com/rs/xyz", "fact": "Riya Sharma asked Prakash Jha to check commission tracking for wishlink.com/rs/xyz." }},
          {{ "relation_type": " CAUSED_ISSUE", "source_entity_name": "Dashboard Update Issue 2025-04-07", "target_entity_name": "wishlink.com/rs/xyz", "fact": "The dashboard update from yesterday caused an issue with commission tracking for wishlink.com/rs/xyz." }}
        ]
        """
    # <<< CHANGE END >>>
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    # No changes needed here for context
    sys_prompt = """You are an AI assistant that determines which facts describing relationships between entities have not been extracted from the given context"""

    user_prompt = f"""
    <PREVIOUS MESSAGES>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </PREVIOUS MESSAGES>
    <CURRENT MESSAGE>
    {context["episode_content"]}
    </CURRENT MESSAGE>

    <EXTRACTED ENTITIES>
    {context["nodes"]}
    </EXTRACTED ENTITIES>

    <EXTRACTED FACTS>
    {context["extracted_facts"]}
    </EXTRACTED FACTS>

    Given the above MESSAGES, list of EXTRACTED ENTITIES, and list of EXTRACTED FACTS describing relationships;
    determine if any significant facts describing relationships *between* the listed entities in the CURRENT MESSAGE haven't been extracted. List the missing facts as sentences.
    """
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


versions: Versions = {'edge': edge, 'reflexion': reflexion}

