from __future__ import annotations

import re
from typing import Any


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(item) for item in value.values())
    return str(value).lower()


def has_any(text: str, *patterns: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def add(tags: set[str], vocabulary: set[str], tag: str) -> None:
    if tag not in vocabulary:
        raise ValueError(f"Tag {tag!r} is not in the controlled vocabulary.")
    tags.add(tag)


def dimension_tags(text: str, vocabulary: set[str]) -> list[str]:
    tags: set[str] = set()

    if has_any(text, r"\b2[-\s]?d\b", r"\btwo[-\s]?dimensional\b", r"\bplanar\b", r"\bslice\b", r"\bscreen[-\s]?space\b", r"\bimage[-\s]?space\b"):
        add(tags, vocabulary, "2d")
    if has_any(text, r"\b3[-\s]?d\b", r"\bthree[-\s]?dimensional\b", r"\bvolume\b", r"\bvolumetric\b"):
        add(tags, vocabulary, "3d")
    if has_any(text, r"\bsurface\b", r"\bmesh\b", r"\btriangular\b", r"\bcurved\b", r"\bmanifold\b"):
        add(tags, vocabulary, "surface")
    if has_any(text, r"\bvolume\b", r"\bvolumetric\b", r"\b3d vector field\b", r"\b3d flow\b"):
        add(tags, vocabulary, "volume")
    if has_any(text, r"\bunsteady\b", r"\btime[-\s]?varying\b", r"\btime[-\s]?dependent\b", r"\b4d\b", r"\btime step\b", r"\bacross time\b", r"\btemporal\b"):
        add(tags, vocabulary, "unsteady")
        add(tags, vocabulary, "time_dependent")
    if has_any(text, r"\bcurvilinear\b"):
        add(tags, vocabulary, "curvilinear_grid")
    if has_any(text, r"\bunstructured\b"):
        add(tags, vocabulary, "unstructured_grid")

    return sorted(tags)


def feature_tags(text: str, vocabulary: set[str]) -> list[str]:
    tags: set[str] = set()

    if has_any(text, r"\bcritical point", r"\bcritical-point", r"\bcritical regions?\b"):
        add(tags, vocabulary, "critical_points")
    if has_any(text, r"\bsaddle"):
        add(tags, vocabulary, "saddles")
    if has_any(text, r"\bsource", r"\bsink"):
        add(tags, vocabulary, "sources_sinks")
    if has_any(text, r"\bvort", r"\bswirl", r"\bcirculation", r"\bcircular flow", r"\beddy\b", r"\bhelical\b"):
        add(tags, vocabulary, "vortices")
    if has_any(text, r"\bvortex core", r"\bvortex-core", r"\bcore line", r"\bvortex lines?\b"):
        add(tags, vocabulary, "vortex_cores")
    if has_any(text, r"\bentropy", r"\bhigh[-\s]?entropy", r"\binformation[-\s]?rich\b"):
        add(tags, vocabulary, "high_entropy_regions")
    if has_any(text, r"\bsalient", r"\bimportant flow", r"\binteresting flow", r"\brepresentative", r"\bfeature"):
        add(tags, vocabulary, "salient_features")
    if has_any(text, r"\btopolog", r"\bseparatrix", r"\bseparatrices", r"\bjacobian", r"\beigen"):
        add(tags, vocabulary, "flow_topology")
    if has_any(text, r"\bboundar", r"\bwall\b", r"\binlet\b", r"\boutlet\b"):
        add(tags, vocabulary, "boundaries")
    if has_any(text, r"\bblank region", r"\bvoid region", r"\bempty region", r"\buncovered region", r"\bcoverage gap"):
        add(tags, vocabulary, "blank_regions")
    if has_any(text, r"\bocclusion", r"\bocclude", r"\bvisibility", r"\btransparent", r"\bopacity\b"):
        add(tags, vocabulary, "occlusion_regions")

    return sorted(tags)


def method_tags(text: str, vocabulary: set[str]) -> list[str]:
    tags: set[str] = set()

    if has_any(text, r"\bevenly[-\s]?spaced", r"\beven spacing", r"\buniform placement\b"):
        add(tags, vocabulary, "evenly_spaced")
    if has_any(text, r"\bfeature[-\s]?guided", r"\bfeature[-\s]?based", r"\bfeature detector", r"\bflow[-\s]?guided"):
        add(tags, vocabulary, "feature_guided")
    if has_any(text, r"\btopology[-\s]?guided", r"\btopology[-\s]?aware", r"\btopolog"):
        add(tags, vocabulary, "topology_guided")
    if has_any(text, r"\bentropy", r"\binformation[-\s]?theoretic", r"\binformation[-\s]?aware", r"\bconditional entropy"):
        add(tags, vocabulary, "entropy_based")
    if has_any(text, r"\bimage[-\s]?guided", r"\bimage[-\s]?space", r"\bscreen[-\s]?space", r"\bview[-\s]?space"):
        add(tags, vocabulary, "image_guided")
    if has_any(text, r"\bview[-\s]?dependent", r"\bviewpoint", r"\bcamera", r"\bscreen[-\s]?space"):
        add(tags, vocabulary, "view_dependent")
    if has_any(text, r"\btemplate", r"\bdiamond", r"\boctahedral", r"\bseed pattern"):
        add(tags, vocabulary, "template_based")
    if has_any(text, r"\bpoisson"):
        add(tags, vocabulary, "poisson_disk")
    if has_any(text, r"\bfarthest"):
        add(tags, vocabulary, "farthest_point")
    if has_any(text, r"\bcluster", r"\bbundle", r"\bagglomerative", r"\bk[-\s]?means"):
        add(tags, vocabulary, "clustering")
    if has_any(text, r"\bsimilar", r"\bdistance metric", r"\bshape distance", r"\bmatching"):
        add(tags, vocabulary, "similarity_based")
    if has_any(text, r"\boptim", r"\bobjective function", r"\benergy", r"\bgreedy", r"\bsequential"):
        add(tags, vocabulary, "optimization_based")
    if has_any(text, r"\binteractive", r"\buser", r"\bmanual", r"\bsketch", r"\bbrush", r"\bgesture", r"\beye[-\s]?tracking"):
        add(tags, vocabulary, "interactive")
    if has_any(text, r"\bhierarchical", r"\bmultilevel", r"\btree\b"):
        add(tags, vocabulary, "hierarchical")
    if has_any(text, r"\bmultiresolution", r"\bmulti[-\s]?resolution", r"\bcoarse[-\s]?to[-\s]?fine", r"\bcontrol grid"):
        add(tags, vocabulary, "multiresolution")

    return sorted(tags)


def task_tags(text: str, vocabulary: set[str]) -> list[str]:
    tags: set[str] = set()

    if has_any(text, r"\bclutter", r"\boverdraw", r"\bdense", r"\bredundant"):
        add(tags, vocabulary, "reduce_clutter")
    if has_any(text, r"\bsalient", r"\bimportant", r"\binformative", r"\binteresting", r"\brepresentative"):
        add(tags, vocabulary, "preserve_salient_features")
    if has_any(text, r"\bcover", r"\bcoverage", r"\bfill", r"\bblank region", r"\bvoid region", r"\buniform"):
        add(tags, vocabulary, "cover_domain")
    if has_any(text, r"\bhighlight", r"\bemphas", r"\breveal", r"\bcapture", r"\bshow"):
        add(tags, vocabulary, "highlight_features")
    if has_any(text, r"\bcompar", r"\bdifference", r"\bbetween.*flow", r"\bmultiple.*flow"):
        add(tags, vocabulary, "compare_flows")
    if has_any(text, r"\bperception", r"\bperceptual", r"\baesthetic", r"\billustrative", r"\billustrator"):
        add(tags, vocabulary, "improve_perception")
    if has_any(text, r"\bocclusion", r"\bvisibility", r"\btransparent", r"\bopacity"):
        add(tags, vocabulary, "reduce_occlusion")
    if has_any(text, r"\btemporal", r"\bcoherence", r"\bunsteady", r"\btime[-\s]?varying", r"\btime[-\s]?dependent"):
        add(tags, vocabulary, "maintain_temporal_coherence")
    if has_any(text, r"\bexplor", r"\binteractive", r"\buser", r"\bsteer", r"\bprobe"):
        add(tags, vocabulary, "support_exploration")
    if has_any(text, r"\brepresentative", r"\bsample", r"\bsubset", r"\bselect.*streamline", r"\bstreamline selection"):
        add(tags, vocabulary, "generate_representative_streamlines")

    return sorted(tags)


def input_tags(text: str, vocabulary: set[str]) -> list[str]:
    tags: set[str] = set()

    if has_any(text, r"\bvector field", r"\bflow field", r"\bvelocity field"):
        add(tags, vocabulary, "vector_field")
    if has_any(text, r"\bscalar field", r"\bdensity map", r"\bimportance field", r"\bprobability distribution"):
        add(tags, vocabulary, "scalar_field")
    if has_any(text, r"\bentropy"):
        add(tags, vocabulary, "entropy_field")
    if has_any(text, r"\bcritical point", r"\bcritical-point"):
        add(tags, vocabulary, "critical_point_locations")
    if has_any(text, r"\bcritical point type", r"\bcritical-point type", r"\bcenter", r"\bspiral", r"\bsaddle", r"\bsource", r"\bsink"):
        add(tags, vocabulary, "critical_point_types")
    if has_any(text, r"\btopology graph", r"\btopological graph", r"\btopolog", r"\bseparatrix", r"\bseparatrices"):
        add(tags, vocabulary, "topology_graph")
    if has_any(text, r"\buser seed", r"\buser[-\s]?specified", r"\binteractive", r"\bmanual", r"\bsketch", r"\bbrush", r"\bgesture", r"\beye[-\s]?tracking", r"\bfixation"):
        add(tags, vocabulary, "user_seed")
    if has_any(text, r"\bviewpoint", r"\bview[-\s]?dependent", r"\bcamera", r"\bview direction"):
        add(tags, vocabulary, "viewpoint")
    if has_any(text, r"\bimage[-\s]?space", r"\bscreen[-\s]?space", r"\bpixel", r"\brendered image"):
        add(tags, vocabulary, "image_space")
    if has_any(text, r"\bdensity", r"\bnumber of streamlines", r"\bstreamline count", r"\bspacing", r"\bseparation"):
        add(tags, vocabulary, "density_control")
    if has_any(text, r"\bdistance threshold", r"\bminimum distance", r"\bseparating distance", r"\bseparation", r"\bdelta_", r"\bradius"):
        add(tags, vocabulary, "distance_threshold")
    if has_any(text, r"\bdetector", r"\bdetection", r"\bdetect", r"\bclassifier", r"\bidentify"):
        add(tags, vocabulary, "feature_detector")

    return sorted(tags)
