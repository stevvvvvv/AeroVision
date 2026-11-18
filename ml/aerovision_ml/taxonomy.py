"""Taxonomie des défauts.

Trois niveaux, volontairement séparés :

1. Défauts MVTec AD (libellés du dataset, par catégorie de pièce).
2. Taxonomie GÉNÉRIQUE AeroVision (ce que le modèle apprend réellement).
3. Correspondance INDICATIVE vers le vocabulaire de contrôle aéronautique.

Le niveau 3 n'est PAS une équivalence : un « crack » MVTec sur une noisette n'est pas
une fissure au sens d'un manuel de maintenance (AMM/SRM) ou d'un contrôle non destructif
certifié. Voir docs/cadre-aeronautique.md.
"""

from __future__ import annotations

MVTEC_CATEGORIES: list[str] = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

# Taxonomie générique (indices stables = classes YOLO)
GENERIC_DEFECTS: list[str] = [
    "crack",           # fissure / cassure
    "scratch",         # rayure
    "hole",            # trou / perforation / impact ponctuel
    "contamination",   # corps étranger, salissure, colle, huile
    "deformation",     # pliage, écrasement, déformation géométrique
    "missing",         # élément manquant
    "misplacement",    # élément mal positionné / inversé
    "cut",             # coupure
    "color",           # défaut de teinte / décoloration
    "surface_texture", # rugosité, défaut de texture
    "other",           # défaut combiné ou non classable
]

GENERIC_LABELS_FR = {
    "crack": "Fissure / cassure",
    "scratch": "Rayure",
    "hole": "Trou / perforation",
    "contamination": "Contamination / corps étranger",
    "deformation": "Déformation",
    "missing": "Élément manquant",
    "misplacement": "Mauvais positionnement",
    "cut": "Coupure",
    "color": "Décoloration",
    "surface_texture": "Défaut de texture",
    "other": "Autre / combiné",
    "unknown": "Anomalie non typée",
}

MVTEC_TO_GENERIC: dict[str, dict[str, str]] = {
    "bottle": {"broken_large": "crack", "broken_small": "crack", "contamination": "contamination"},
    "cable": {
        "bent_wire": "deformation", "cable_swap": "misplacement", "combined": "other",
        "cut_inner_insulation": "cut", "cut_outer_insulation": "cut",
        "missing_cable": "missing", "missing_wire": "missing", "poke_insulation": "hole",
    },
    "capsule": {"crack": "crack", "faulty_imprint": "other", "poke": "hole",
                "scratch": "scratch", "squeeze": "deformation"},
    "carpet": {"color": "color", "cut": "cut", "hole": "hole",
               "metal_contamination": "contamination", "thread": "contamination"},
    "grid": {"bent": "deformation", "broken": "crack", "glue": "contamination",
             "metal_contamination": "contamination", "thread": "contamination"},
    "hazelnut": {"crack": "crack", "cut": "cut", "hole": "hole", "print": "other"},
    "leather": {"color": "color", "cut": "cut", "fold": "deformation",
                "glue": "contamination", "poke": "hole"},
    "metal_nut": {"bent": "deformation", "color": "color", "flip": "misplacement", "scratch": "scratch"},
    "pill": {"color": "color", "combined": "other", "contamination": "contamination",
             "crack": "crack", "faulty_imprint": "other", "pill_type": "other", "scratch": "scratch"},
    "screw": {"manipulated_front": "deformation", "scratch_head": "scratch", "scratch_neck": "scratch",
              "thread_side": "deformation", "thread_top": "deformation"},
    "tile": {"crack": "crack", "glue_strip": "contamination", "gray_stroke": "color",
             "oil": "contamination", "rough": "surface_texture"},
    "toothbrush": {"defective": "other"},
    "transistor": {"bent_lead": "deformation", "cut_lead": "cut",
                   "damaged_case": "deformation", "misplaced": "misplacement"},
    "wood": {"color": "color", "combined": "other", "hole": "hole",
             "liquid": "contamination", "scratch": "scratch"},
    "zipper": {"broken_teeth": "crack", "combined": "other", "fabric_border": "surface_texture",
               "fabric_interior": "surface_texture", "rough": "surface_texture",
               "split_teeth": "deformation", "squeezed_teeth": "deformation"},
}

# Pertinence de chaque catégorie MVTec comme *proxy* d'une pièce aéronautique.
# Jugement qualitatif, documenté dans docs/donnees.md.
AERO_PROXY_RELEVANCE: dict[str, str] = {
    "metal_nut": "élevée — fixation métallique usinée",
    "screw": "élevée — visserie / filetage",
    "grid": "moyenne — structure métallique ajourée",
    "cable": "moyenne — harnais électrique",
    "transistor": "moyenne — composant électronique (avionique)",
    "tile": "faible — matériau céramique plan",
    "bottle": "faible",
    "capsule": "faible",
    "carpet": "faible — textile (aménagement cabine au mieux)",
    "hazelnut": "nulle",
    "leather": "faible — revêtement (cabine)",
    "pill": "nulle",
    "toothbrush": "nulle",
    "wood": "nulle",
    "zipper": "nulle",
}

# Correspondance INDICATIVE générique -> vocabulaire de contrôle aéro.
# Sert uniquement à orienter l'opérateur ; la qualification reste humaine et documentaire.
GENERIC_TO_AERO_HINT: dict[str, str] = {
    "crack": "Indication linéaire possible (fissure) — à confirmer par CND (ressuage, courants de Foucault…)",
    "scratch": "Rayure / marque d'outil — évaluer profondeur vs tolérances SRM",
    "hole": "Impact / piqûre / perforation — évaluer dimension, rechercher corrosion",
    "contamination": "Corps étranger / FOD / résidu — nettoyage puis ré-inspection",
    "deformation": "Bosse / déformation — mesure dimensionnelle requise",
    "missing": "Élément manquant (frein, goupille, rivet…) — non-conformité d'assemblage",
    "misplacement": "Erreur de montage — vérification documentaire",
    "cut": "Coupure / entaille — évaluer profondeur",
    "color": "Décoloration — possible surchauffe ou dégradation de revêtement",
    "surface_texture": "Dégradation de surface / revêtement",
    "other": "Anomalie à qualifier par l'opérateur",
    "unknown": "Anomalie non typée détectée par écart à la normale — qualification humaine",
}


def to_generic(category: str, mvtec_defect: str) -> str:
    """Convertit un libellé de défaut MVTec en classe générique."""
    if mvtec_defect == "good":
        raise ValueError("'good' n'est pas un défaut")
    return MVTEC_TO_GENERIC.get(category, {}).get(mvtec_defect, "other")


def generic_index(name: str) -> int:
    return GENERIC_DEFECTS.index(name)
