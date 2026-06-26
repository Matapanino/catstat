"""Output column metadata and feature-name construction.

Column order is feature-major, then stat order, then (for class-expanded multiclass stats) class
order. The same metadata list drives both ``transform`` assembly and ``get_feature_names_out`` so
they can never disagree.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._stats import StatSpec


@dataclass(frozen=True)
class ColumnMeta:
    feature: object
    stat: str
    class_label: object  # None unless a class-expanded multiclass column
    target_dependent: bool
    name: str


def build_columns(cat_cols, specs: list[StatSpec], target_type, classes) -> list[ColumnMeta]:
    cols: list[ColumnMeta] = []
    for feat in cat_cols:
        for spec in specs:
            if spec.class_expanded and target_type == "multiclass":
                for c in classes:
                    cols.append(
                        ColumnMeta(
                            feature=feat,
                            stat=spec.name,
                            class_label=c,
                            target_dependent=spec.target_dependent,
                            name=f"{feat}__{spec.name_infix}__class_{c}",
                        )
                    )
            else:
                cols.append(
                    ColumnMeta(
                        feature=feat,
                        stat=spec.name,
                        class_label=None,
                        target_dependent=spec.target_dependent,
                        name=f"{feat}__{spec.name_infix}",
                    )
                )
    return cols
