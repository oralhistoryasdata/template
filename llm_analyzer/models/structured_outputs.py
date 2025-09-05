from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class QualityScore(BaseModel):
    score: int = Field(..., ge=0, le=1, description="1 if summary is good, 0 if not")
    reasoning: str = Field(..., description="Brief explanation for the score")

class UtteranceGroup(BaseModel):
    """Group of utterances from transcript rows"""
    utterances: List[str] = Field(..., description="List of significant utterances")
    context: str = Field(..., description="Brief context about this group")

class ThematicTerms(BaseModel):
    """Generated thematic terms from utterances"""
    terms: List[str] = Field(..., description="List of thematic terms generated")

class FilterEntry(BaseModel):
    """Single entry for the gen-filters.csv file"""
    tag: str = Field(..., description="Short tag identifier")
    description: str = Field(..., description="Description of the tag")

class GeneratedFilters(BaseModel):
    """Complete set of generated filters"""
    filters: List[FilterEntry] = Field(..., description="List of filter entries")

class RowTags(BaseModel):
    """Tags for a single CSV row"""
    row_index: int = Field(..., description="Index of the row in CSV")
    gen_tags: str = Field(..., description="Semicolon-separated tags for this row")

class TranscriptTags(BaseModel):
    """Tags for an entire transcript"""
    filename: str = Field(..., description="Name of the transcript file")
    row_tags: List[RowTags] = Field(..., description="Tags for each row")

class QualityValidation(BaseModel):
    """Validation of generated tags quality"""
    is_valid: bool = Field(..., description="Whether tags are of good quality")
    feedback: str = Field(..., description="Feedback on tag quality")