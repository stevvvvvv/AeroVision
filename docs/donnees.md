# Données

## MVTec AD (dataset principal)

- 15 catégories (5 textures, 10 objets), ~5 350 images haute résolution, masques de défauts
  au pixel près.
- Entraînement : images **saines uniquement** (`train/good`). Test : saines + défectueuses.
- **Licence CC BY-NC-SA 4.0 : usage commercial interdit.** Ce projet est un démonstrateur de
  portfolio. Un produit commercial devra être entraîné sur d'autres données (propriétaires ou
  sous licence permissive).
- Téléchargement : <https://www.mvtec.com/company/research/datasets/mvtec-ad> (formulaire),
  puis extraire dans `data/mvtec_ad/`.

### Pertinence de chaque catégorie comme proxy aéronautique

| Catégorie | Proxy aéro | Pourquoi |
|---|---|---|
| metal_nut | élevée | fixation métallique usinée, rayures, déformations, inversion de montage |
| screw | élevée | visserie, défauts de filetage et de tête |
| grid | moyenne | structure métallique ajourée |
| cable | moyenne | harnais : coupures d'isolant, fils manquants, inversion |
| transistor | moyenne | composant électronique (avionique) : pattes pliées, mauvais placement |
| tile | faible | céramique plane |
| autres | faible à nulle | noisette, pilule, brosse à dents, bois… utiles seulement pour le benchmark |

Pour la démo recruteur : mettre en avant **metal_nut, screw, cable, transistor, grid**.

### Protocole d'évaluation (honnête)

MVTec AD n'a pas de jeu de validation avec défauts. Le script `train_patchcore.py` :

1. retire 10 % des images saines d'entraînement de la mémoire (statistiques « normales ») ;
2. coupe le test en **calibration (30 %)** et **évaluation (70 %)**, stratifié par type de défaut ;
3. calibre les seuils et le score de confiance sur la calibration uniquement ;
4. rapporte les métriques sur l'évaluation uniquement (liste des fichiers stockée dans le modèle).

Les chiffres ne sont donc pas directement comparables aux AUROC « test complet » des articles,
mais ils ne sont pas gonflés par une calibration sur les données d'évaluation.

## Datasets complémentaires

Tous ont des licences à vérifier avant usage ; elles conditionnent la monétisation.

| Dataset | Contenu | Intérêt pour AeroVision | Licence (à vérifier) |
|---|---|---|---|
| MVTec AD 2 | 8 scénarios, éclairages variables | robustesse aux conditions d'acquisition | non commerciale (MVTec) |
| VisA (Amazon) | 12 objets dont PCB, ~10 800 images | électronique, multi-instances | CC BY 4.0 sur le registre AWS, mentionnée CC BY-NC-SA 4.0 par anomalib : vérifier à la source |
| NEU Surface Defect | défauts de surface d'acier laminé | métal, 6 classes supervisées | usage académique |
| GDXray (castings) | radiographies de pièces moulées | se rapproche du CND radiographique | usage recherche |
| Jeux « aircraft surface defect » communautaires (Roboflow…) | photos de fuselage (corrosion, bosses, rivets manquants) | seul contenu réellement aéro | variable, qualité d'annotation inégale |

Le format MVTec (`train/good`, `test/<défaut>`, `ground_truth/<défaut>`) est réutilisable :
tout dataset converti dans ce format passe dans le pipeline sans changer le code.

## Images redistribuées dans la démo statique

`frontend/public/demo/` contient une trentaine d'images MVTec AD (redimensionnées) et leurs
dérivés (cartes d'anomalie, superpositions, masques). Elles restent sous **CC BY-NC-SA 4.0**,
avec attribution dans `frontend/public/demo/LICENSE-MVTEC.txt` et dans le bandeau de la démo.
Le dataset complet n'est jamais versionné (`data/` est dans `.gitignore`).

## Données synthétiques

`ml/scripts/make_synthetic_dataset.py` génère des plaques métalliques brossées avec rivets et
défauts dessinés (fissure, rayure, trou, contamination). Usage **exclusif** : tests
automatisés et démo hors-ligne. Aucune métrique obtenue dessus n'a de valeur.

## Licences logicielles

- Code AeroVision : MIT.
- PyTorch / torchvision : BSD. Poids ImageNet torchvision : voir les conditions de chaque modèle.
- **Ultralytics YOLO : AGPL-3.0.** Utilisation optionnelle (typage des défauts). Pour un
  produit fermé : licence commerciale Ultralytics ou détecteur sous licence permissive.
