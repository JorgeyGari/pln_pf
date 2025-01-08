import ast
import json
from pydantic import BaseModel, ValidationError
from typing import List, Union

# Pydantic data models
class PageSection(BaseModel):
    """Data model for a Wikipedia page section."""
    name: str
    sections: List[str]

class LLMOutput(BaseModel):
    data: List[PageSection]


def validate_info_output(info_output: str) -> LLMOutput:
    """
    Validates the LLM's info output against the Pydantic schema.
    Args:
        info_output (str): The raw output string from the LLM.
    Returns:
        LLMOutput: A validated Pydantic model instance.
    Raises:
        ValueError: If the output does not conform to the schema.
    """
    try:
        # Safely parse the input string into a Python object
        raw_data = json.loads(info_output)  # Use json.loads if input is JSON
    except json.JSONDecodeError:
        try:
            raw_data = ast.literal_eval(info_output)  # Fallback to literal_eval
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid format in LLM output: {e}\nRAW\n{info_output}")

    # Ensure data structure matches expectations
    if not isinstance(raw_data, list):
        raise ValueError(f"LLM output must be a list of sections, but is: {type(raw_data)}.\nRAW\n{raw_data}")

    try:
        # Map raw data into the PageSection model
        validated_data = LLMOutput(data=[
            PageSection(
                name=item.get("name"),  # Use .get() for safety
                sections=item.get("sections", [])
            )
            for item in raw_data
        ])
        return validated_data
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")
