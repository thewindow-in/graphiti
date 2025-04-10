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

MODIFIED FOR CLARITY ON DEFAULTING TO REFERENCE TIMESTAMP
"""

from typing import Any, Optional, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class EdgeDates(BaseModel):
    valid_at: Optional[str] = Field(
        None,
        description='The date and time when the relationship described by the edge fact became true or was established. Format: YYYY-MM-DDTHH:MM:SS.ffffffZ or null.',
    )
    invalid_at: Optional[str] = Field(
        None,
        description='The date and time when the relationship described by the edge fact stopped being true or ended. Format: YYYY-MM-DDTHH:MM:SS.ffffffZ or null.',
    )


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    # <<< CHANGE START >>>
    # Updated system prompt and guidelines for clarity on timestamp defaulting
    return [
        Message(
            role='system',
            content='You are an AI assistant that extracts specific validity start and end datetimes for graph edge relationships based on conversation context. You prioritize explicit dates mentioned in the text relating *directly* to the fact itself.',
        ),
        Message(
            role='user',
            content=f"""
            <PREVIOUS MESSAGES>
            {context['previous_episodes']}
            </PREVIOUS MESSAGES>
            <CURRENT MESSAGE>
            {context["current_episode"]}
            </CURRENT MESSAGE>
            <REFERENCE TIMESTAMP>
            {context['reference_timestamp']}
            </REFERENCE TIMESTAMP>

            <FACT>
            {context['edge_fact']}
            </FACT>

            Definitions:
            - valid_at: The specific date and time the relationship described by the FACT became true or was established.
            - invalid_at: The specific date and time the relationship described by the FACT stopped being true or ended.

            Task:
            Analyze the FACT and the conversation context (CURRENT MESSAGE and PREVIOUS MESSAGES). Determine the `valid_at` and `invalid_at` datetimes for the relationship described in the FACT.

            Guidelines:
            1.  **Prioritize Explicit Dates:** Only extract `valid_at` or `invalid_at` if a specific date or relative time (e.g., "yesterday", "last week", "since March 1st", "ended on Tuesday") is mentioned *directly in relation to the FACT itself* within the CURRENT or PREVIOUS messages.
            2.  **Calculate Relative Times:** If a relative time directly related to the FACT is found (e.g., "The issue started 2 hours ago", "They became partners last month"), calculate the absolute datetime based on the REFERENCE TIMESTAMP ({context['reference_timestamp']}).
            3.  **Default `valid_at` to Reference Time:** If NO explicit start date/time for the FACT can be determined from the context, BUT the fact seems to be true *at the time of the CURRENT MESSAGE* (e.g., stated in present tense, describes a current state), set `valid_at` to the REFERENCE TIMESTAMP ({context['reference_timestamp']}).
            4.  **Leave Null if Uncertain:** If you cannot determine a specific start or end time based on explicit mentions or the default rule above, leave `valid_at` and/or `invalid_at` as null. Do not guess or infer dates from unrelated events.
            5.  **Format:** Use ISO 8601 format (YYYY-MM-DDTHH:MM:SS.ffffffZ). Use Z for UTC. If only a date is mentioned, use 00:00:00 for the time. If only a year is mentioned, use January 1st, 00:00:00.
            6.  **Spanning Relationships:** If a relationship has both a start and end time explicitly mentioned (e.g., "The campaign ran from Jan 1st to Jan 31st"), extract both `valid_at` and `invalid_at`.
            7.  **Point-in-Time Facts:** If a fact represents a state at a specific time without an explicit end (e.g., "He joined the team on Monday"), only extract `valid_at`.
            """,
        ),
    ]
    # <<< CHANGE END >>>


versions: Versions = {'v1': v1}

