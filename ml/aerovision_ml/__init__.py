"""AeroVision ML — détection d'anomalies visuelles sur pièces industrielles.

Modules :
- taxonomy     : catégories MVTec AD, taxonomie générique des défauts, correspondance indicative aéro
- data         : chargement du dataset MVTec AD (et de tout dataset au même format)
- patchcore    : détection/segmentation d'anomalies non supervisée (PatchCore)
- classifier   : classification du type de pièce (routage vers le bon modèle PatchCore)
- postprocess  : carte d'anomalie -> masque -> régions localisées (OpenCV)
- calibration  : seuils et score de confiance calibrés
- metrics      : AUROC image/pixel, F1
- baseline     : détecteur OpenCV classique (sans IA), utilisé si aucun modèle n'est entraîné
"""

__version__ = "0.1.0"
