from __future__ import annotations

from pydantic import BaseModel, model_validator
from qdrant_client.http import models as qmodels


class MetadataFilter(BaseModel):
    """Filter applied against indexed chunk metadata stored in Qdrant."""

    filename: str | None = None
    filenames: list[str] | None = None
    page: int | None = None
    section: str | None = None
    document_id: str | None = None

    @model_validator(mode="after")
    def _normalize(self) -> "MetadataFilter":
        # Normalize filenames list: strip blanks, deduplicate order
        names = [n.strip() for n in (self.filenames or []) if isinstance(n, str) and n.strip()]

        if not names:
            self.filenames = None
        elif len(names) == 1:
            self.filename, self.filenames = names[0], None
        else:
            self.filename, self.filenames, self.page = None, names, None

        if self.filename is not None:
            self.filename = self.filename.strip() or None
        if self.section is not None:
            self.section = self.section.strip() or None
        if self.document_id is not None:
            self.document_id = self.document_id.strip() or None

        return self


# Coercion helpers

def coerce_filter(
    filters: MetadataFilter | dict[str, object] | None,
) -> MetadataFilter | None:
    """Coerce a raw dict (or None) into a normalised `MetadataFilter`."""
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        return filters
    if isinstance(filters, dict):
        return MetadataFilter.model_validate(filters)
    raise TypeError(f"Unsupported filters type: {type(filters).__name__}")


def filters_to_dict(
    filters: MetadataFilter | dict[str, object] | None,
) -> dict[str, object] | None:
    """Return a flat dict suitable for logging or downstream processing."""
    f = coerce_filter(filters)
    if f is None:
        return None
    return f.model_dump(exclude_none=True) or None


def filters_to_qdrant(
    filters: MetadataFilter | dict[str, object] | None,
) -> qmodels.Filter | None:
    
    flat = filters_to_dict(filters)
    if not flat:
        return None

    conditions: list[qmodels.FieldCondition] = []

    for field, value in flat.items():
        if value is None:
            continue

        # Multi-filename → MatchAny
        if field == "filenames" and isinstance(value, list):
            valid_names = [x for x in value if isinstance(x, str) and x]
            if valid_names:
                conditions.append(
                    qmodels.FieldCondition(
                        key="metadata.filename",
                        match=qmodels.MatchAny(any=valid_names),
                    )
                )
            continue

        # Scalar string / int → MatchValue
        if isinstance(value, (str, int)):
            conditions.append(
                qmodels.FieldCondition(
                    key=f"metadata.{field}",
                    match=qmodels.MatchValue(value=value),
                )
            )

    return qmodels.Filter(must=conditions) if conditions else None
