from __future__ import annotations


def get_toy_samples():
    """
    Return a list of small toy samples for geometry demo.
    Boxes are normalized [0,1].

    objects: list of {label, box}
    relations: list of {subj, obj, pred} (pred is placeholder for now)
    """
    samples = []

    # 1) Overlap case
    samples.append({
        "name": "overlap",
        "prompt": "two objects overlapping",
        "objects": [
            {"label": "A", "box": [0.10, 0.10, 0.60, 0.60]},
            {"label": "B", "box": [0.40, 0.40, 0.90, 0.90]},
        ],
        "relations": [{"subj": 0, "obj": 1, "pred": "interact"}]
    })

    # 2) X-overlap only (stacked vertically)
    samples.append({
        "name": "x_overlap_only",
        "prompt": "objects aligned in x, separated in y",
        "objects": [
            {"label": "A", "box": [0.20, 0.10, 0.80, 0.30]},
            {"label": "B", "box": [0.30, 0.70, 0.70, 0.90]},
        ],
        "relations": [{"subj": 0, "obj": 1, "pred": "interact"}]
    })

    # 3) Y-overlap only (side by side)
    samples.append({
        "name": "y_overlap_only",
        "prompt": "objects aligned in y, separated in x",
        "objects": [
            {"label": "A", "box": [0.10, 0.30, 0.30, 0.80]},
            {"label": "B", "box": [0.70, 0.40, 0.90, 0.70]},
        ],
        "relations": [{"subj": 0, "obj": 1, "pred": "interact"}]
    })

    # 4) Fully separated (diagonal)
    samples.append({
        "name": "separated",
        "prompt": "objects separated diagonally",
        "objects": [
            {"label": "A", "box": [0.10, 0.10, 0.25, 0.25]},
            {"label": "B", "box": [0.70, 0.70, 0.90, 0.90]},
        ],
        "relations": [{"subj": 0, "obj": 1, "pred": "interact"}]
    })

    # 5) Containment (B inside A)
    samples.append({
        "name": "containment",
        "prompt": "one object contains another",
        "objects": [
            {"label": "A", "box": [0.10, 0.10, 0.90, 0.90]},
            {"label": "B", "box": [0.35, 0.35, 0.60, 0.60]},
        ],
        "relations": [{"subj": 0, "obj": 1, "pred": "contain"}]
    })

    return samples
