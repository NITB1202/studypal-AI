from enum import Enum

class PreprocessStrategy(Enum):
    DIRECT = "direct"
    CHUNK = "chunk"
    SUMMARIZE = "summarize"