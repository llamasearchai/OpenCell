import abc
from typing import Any, Dict, List, Optional

class DistributedTaskManager(abc.ABC):
    """Abstract base class for a distributed task manager."""

    @abc.abstractmethod
    async def submit_task(self, task_payload: Dict[str, Any], resource_requirements: Optional[Dict[str, Any]] = None) -> str:
        """Submit a task for distributed execution."""
        pass

    @abc.abstractmethod
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a distributed task."""
        pass

    @abc.abstractmethod
    async def get_task_statuses(self, task_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get statuses for multiple tasks."""
        pass

    @abc.abstractmethod
    async def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """Get the result of a completed distributed task."""
        pass

    @abc.abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a distributed task."""
        pass 