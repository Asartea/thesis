from dataclasses import dataclass

from data_generation.config import MODEL


@dataclass(frozen=True)
class Job:
    year: int
    day: int
    prompt: str
    code_variant: str
    style_variant: str

    @property
    def id(self) -> str:
        return (
            f"{MODEL}-{self.year}-{self.day}-{self.code_variant}-{self.style_variant}"
        )
