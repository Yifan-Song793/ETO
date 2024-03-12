import os
import json
import logging
import functools
from abc import ABC
from typing import List, Optional, Tuple, Any


logger = logging.getLogger("agent_frame")


class Task(ABC):
    """Base class for a task instance."""

    task_name: str = "base"

    def __init__(self, **kwargs) -> None:
        self.task_id: Any = kwargs.get("task_id", None)
        
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int) -> Tuple[List["Task"], int]:
        pass
