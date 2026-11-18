# Défauts industriels génériques vs défauts aéronautiques certifiés

Ce document existe parce que la distinction est la première question qu'un ingénieur qualité
de Safran, Airbus Atlantic, Thales ou Daher posera. AeroVision y répond sans ambiguïté.

## Ce que fait AeroVision

AeroVision détecte, localise et segmente des **écarts visuels à la normale** sur des images de
pièces, puis propose un **type de défaut générique** (fissure, rayure, contamination…). Il est
entraîné et évalué sur des datasets industriels publics (MVTec AD) dont aucune image ne
représente une pièce aéronautique réelle.

## Ce qu'AeroVision ne fait pas

| Exigence aéronautique | Situation dans AeroVision |
|---|---|
| Défaut défini par rapport à une **tolérance documentée** (plan, SRM, AMM, CMM, spécification matériau) | Non : le modèle détecte un écart statistique, sans notion de cote ni de tolérance |
| Contrôle réalisé par du **personnel CND certifié** (EN 4179 / NAS 410, niveaux 1-2-3) | Non : outil d'aide, la qualification reste humaine |
| Méthodes **CND qualifiées** (ressuage, magnétoscopie, courants de Foucault, ultrasons, radiographie, thermographie) | Non : imagerie visible uniquement. Un défaut sous-surfacique ou une fissure fermée sont invisibles |
| **Probabilité de détection démontrée** (POD, typiquement a90/95 sur éprouvettes à défauts calibrés) | Non : les métriques MVTec (AUROC) ne sont pas une POD |
| Logiciel qualifié (DO-178C si embarqué ; procédures qualité du Part-21 / Part-145 au sol) | Non : démonstrateur, sans dossier de qualification |
| Décision de **remise en service / libération** de pièce | Jamais : décision humaine signée, tracée |

## Correspondance générique → aéronautique

La table `GENERIC_TO_AERO_HINT` (ml/aerovision_ml/taxonomy.py) associe chaque classe générique
à une **piste** de vocabulaire aéro affichée à l'opérateur (ex. « crack » → « indication
linéaire possible, à confirmer par CND »). C'est une aide à l'orientation, pas une
classification réglementaire.

Exemples de défauts aéronautiques **absents** de MVTec AD et donc non appris :

- corrosion (piqûres, exfoliante, filiforme) ;
- délaminage, porosité, impact basse énergie sur composite (BVID) ;
- criques de fatigue autour de fixations ;
- brûlures / surchauffe sur aubes de turbine, érosion de bord d'attaque ;
- arrachement de peinture ou de revêtement anti-érosion ;
- FOD (Foreign Object Damage) sur aubes de soufflante.

## Comment le projet gère ce risque

1. **Validation humaine obligatoire** : toutes les inspections entrent dans une file de revue,
   y compris « conforme probable ». Le verdict du modèle ne sert qu'à prioriser.
2. **Indicateur critique mis en avant** : le taux de défauts manqués (faux négatifs), pas la
   précision globale. Dans l'aéro, un faux négatif coûte infiniment plus qu'une fausse alarme.
3. **Détection non supervisée** (PatchCore) : le modèle apprend la pièce saine et signale tout
   écart, y compris des défauts jamais vus. C'est la bonne posture quand les défauts réels
   sont rares et confidentiels.
4. **Traçabilité** : revues en ajout seul, opérateur, date, version du modèle, verdict du
   modèle au moment de la revue.
5. **Boucle de retour** : les fausses alarmes et défauts manqués signalés par les opérateurs
   sont exportés (CSV) pour ré-entraîner et recalibrer.
6. **Transparence sur le moteur** : l'interface affiche si le résultat vient d'un modèle
   entraîné ou du repli OpenCV sans IA, et si la calibration est fiable.

## Ce qu'il faudrait pour aller vers un usage réel

- Données propriétaires d'un industriel (images de pièces réelles, défauts annotés par des
  contrôleurs certifiés) et un protocole d'acquisition maîtrisé (éclairage, optique, pose).
- Une étude POD sur éprouvettes à défauts calibrés, avec l'opérateur dans la boucle.
- Un positionnement clair : **aide au contrôleur** (tri, priorisation, seconde lecture) plutôt
  que remplacement du contrôle.
- Un dossier de qualification logiciel conforme aux procédures qualité de l'organisme
  (Part-21 conception/production, Part-145 maintenance) et, le cas échéant, l'accord de
  l'autorité.
