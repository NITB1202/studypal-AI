from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class RAGChunk:
    id: str
    content: str
    metadata: Dict[str, Any]
