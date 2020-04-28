from .interface import PlatformInterface


class Labelbox(PlatformInterface):
    def fetch_jobs(self, status: str, limit: int):
        pass

    def fetch_annotations(self, job_name: str):
        pass
