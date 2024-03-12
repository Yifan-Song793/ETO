import os
import json
import yaml
import logging
from typing import Iterable, List, Tuple

from scienceworld import ScienceWorldEnv

from eval_agent.tasks.base import Task

logger = logging.getLogger("agent_frame")


class SciWorldTask(Task):
    """ScienceWorld task instance."""
    
    task_name = "sciworld"
    
    def __init__(
        self,
        sub_task_name: str,
        variation_idx: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name
        self.variation_idx = variation_idx
    
    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1):
        if split == 'train':
            task_idxs = json.load(open("eval_agent/data/sciworld/train_indices.json"))
        elif split == 'dev':
            task_idxs = json.load(open("eval_agent/data/sciworld/dev_indices.json"))
        elif split == 'test':
            task_idxs = json.load(open("eval_agent/data/sciworld/test_indices.json"))
        else:
            raise ValueError
        taskname2id = json.load(open("eval_agent/data/sciworld/taskname2id.json"))
        if part_num == 1:
            task_idxs = task_idxs
        else:
            assert part_idx != -1
            part_len = len(task_idxs) // part_num + 1
            task_idxs = task_idxs[part_len * part_idx: part_len * (part_idx + 1)]
        N_TASKS = len(task_idxs)
        
        def generator():
            for item in task_idxs:
                task_name = item[0]
                variation_idx = item[1]
                yield cls(
                    task_id=f"{taskname2id[task_name]}_{variation_idx}",
                    sub_task_name=task_name,
                    variation_idx=variation_idx,
                )
                    
        return generator(), N_TASKS
    