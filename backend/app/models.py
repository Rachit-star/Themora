from dataclasses import dataclass, field, asdict


@dataclass
class Finding:
    detector: str
    severity: str
    column: str | None
    title: str
    evidence: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)