from enum import Enum


class JobStatus(str, Enum):
    """Job statuses that will be returned by the API & SDK"""

    UNKNOWN = "UNKNOWN"
    QUEUED = "QUEUED"  # Job is waiting to start
    STARTING = "STARTING"  # Job is in the process of starting up
    RUNNING = "RUNNING"  # Job is actively running
    RECOVERING = "RECOVERING"  # Keep polling the same job ID
    SUCCEEDED = "SUCCEEDED"  # Job completed successfully
    CANCELLING = "CANCELLING"  # Job is in the process of being canceled
    CANCELLED = "CANCELLED"  # Job was canceled by the user
    FAILED = "FAILED"  # Job failed

    @classmethod
    def terminal_statuses(cls) -> list["JobStatus"]:
        return [
            cls.SUCCEEDED,
            cls.FAILED,
            cls.CANCELLING,
            cls.CANCELLED,
        ]

    def is_terminal(self) -> bool:
        return self in self.terminal_statuses()
