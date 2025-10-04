"""Dataclasses representing intermediate legal structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SubpointNode:
    label: Optional[str]
    text: str
    start: int
    end: int


@dataclass
class PointNode:
    label: Optional[str]
    text: str
    start: int
    end: int
    subpoints: List[SubpointNode] = field(default_factory=list)


@dataclass
class PartNode:
    label: Optional[str]
    text: str
    start: int
    end: int
    points: List[PointNode] = field(default_factory=list)


@dataclass
class ArticleNode:
    label: Optional[str]
    title: Optional[str]
    start: int
    end: int
    parts: List[PartNode] = field(default_factory=list)


@dataclass
class SectionNode:
    label: Optional[str]
    title: Optional[str]
    start: int
    end: int
    articles: List[ArticleNode] = field(default_factory=list)


@dataclass
class ChapterNode:
    label: Optional[str]
    title: Optional[str]
    start: int
    end: int
    sections: List[SectionNode] = field(default_factory=list)


@dataclass
class DivisionNode:
    label: Optional[str]
    title: Optional[str]
    start: int
    end: int
    chapters: List[ChapterNode] = field(default_factory=list)


@dataclass
class DocumentStructure:
    divisions: List[DivisionNode] = field(default_factory=list)
    chapters: List[ChapterNode] = field(default_factory=list)
    sections: List[SectionNode] = field(default_factory=list)
    articles: List[ArticleNode] = field(default_factory=list)


__all__ = [
    "ArticleNode",
    "ChapterNode",
    "DivisionNode",
    "DocumentStructure",
    "PartNode",
    "PointNode",
    "SectionNode",
    "SubpointNode",
]
