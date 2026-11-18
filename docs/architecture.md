# Architecture

## Chaîne d'inspection

```mermaid
flowchart TD
    A[Image de la pièce<br/>caméra ou upload] --> B[FastAPI<br/>POST /api/inspections]
    B --> C{Classifieur CNN<br/>ResNet-18}
    C -->|type de pièce| D{Modèle PatchCore<br/>pour cette pièce ?}
    D -->|oui| E[PatchCore<br/>WideResNet-50 + memory bank]
    D -->|non| F[Repli OpenCV<br/>sans IA, signalé]
    E --> G[Carte d'anomalie<br/>+ score image]
    F --> G
    G --> H[Post-traitement OpenCV<br/>seuil, morphologie, contours]
    H --> I[Régions localisées<br/>boîtes, polygones, surface]
    I --> J{YOLO-seg<br/>disponible ?}
    J -->|oui| K[Typage du défaut<br/>taxonomie générique]
    J -->|non| L[Anomalie non typée]
    K --> M[Calibration<br/>confiance = P défaut]
    L --> M
    M --> N[Verdict de priorisation<br/>conforme probable / à vérifier / non conforme probable]
    N --> O[(Base SQL<br/>inspections, détections, revues)]
    N --> P[Stockage images<br/>original, heatmap, overlay, masque]
    O --> Q[React<br/>file de validation]
    Q --> R[Revue opérateur<br/>confirme, rejette, ajoute, décide]
    R --> O
    O --> S[Export CSV<br/>ré-entraînement]
```

## Pourquoi deux familles de modèles

```mermaid
flowchart TD
    A[Défauts réels rares,<br/>confidentiels, variés] --> B[PatchCore non supervisé<br/>apprend la pièce saine]
    B --> C[Détecte tout écart,<br/>y compris défaut jamais vu]
    D[Quelques défauts annotés] --> E[YOLO supervisé<br/>apprend des classes]
    E --> F[Nomme le défaut,<br/>mais rate l'inconnu]
    C --> G[Fusion : PatchCore localise,<br/>YOLO qualifie]
    F --> G
```

## Modèle de données

```mermaid
flowchart TD
    I[Inspection<br/>id, date, référence pièce, catégorie,<br/>moteur, version modèle, score, confiance,<br/>verdict, statut, décision finale] --> D[Detection<br/>rang, boîte, polygone, surface,<br/>score max/moyen, type, source]
    I --> R[Review — ajout seul<br/>opérateur, date, décision, commentaire,<br/>retour par détection, zones ajoutées,<br/>verdict modèle au moment de la revue]
```

## API

| Méthode | Route | Rôle |
|---|---|---|
| GET | `/api/health` | sonde |
| GET | `/api/system` | modèles chargés, politique de décision, classes |
| POST | `/api/inspections` | inspecter une image (multipart : `file`, `part_reference`, `category`) |
| GET | `/api/inspections` | historique / file (`status`, `verdict`, `category`, `queue=true`) |
| GET | `/api/inspections/{id}` | détail complet |
| GET | `/api/inspections/{id}/image/{original\|heatmap\|overlay\|mask}` | images |
| POST | `/api/inspections/{id}/reviews` | validation humaine |
| GET | `/api/stats` | KPI, concordance modèle / opérateur |
| GET | `/api/export/feedback.csv` | retours opérateurs pour ré-entraînement |

Documentation interactive : `http://localhost:8000/docs`.

## Choix techniques

- **PatchCore plutôt qu'un autoencodeur** : état de l'art robuste sur MVTec AD, pas
  d'entraînement de réseau (seulement extraction + mémoire), donc rapide et reproductible.
- **Un modèle par type de pièce** : reflète la réalité industrielle (chaque référence a sa
  normalité) et garde des mémoires petites.
- **Bornes de normalisation fixes** pour la carte de chaleur : une pièce saine reste « froide »
  au lieu d'être étirée sur toute la palette (piège classique du min/max par image).
- **Revues en ajout seul** : exigence de traçabilité.
- **SQLite par défaut, PostgreSQL par variable d'environnement** (`AEROVISION_DATABASE_URL`).
