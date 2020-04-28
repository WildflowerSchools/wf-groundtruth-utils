import abc


class PlatformInterface(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'fetch_jobs') and
                callable(subclass.fetch_jobs) and
                hasattr(subclass, 'fetch_annotations') and
                callable(subclass.fetch_annotations) or
                hasattr(subclass, 'generate_image_set') and
                callable(subclass.generate_image_set) or
                NotImplemented)

    @abc.abstractmethod
    def fetch_jobs(self, status: str, limit: int):
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_annotations(self, job_name: str):
        raise NotImplementedError
