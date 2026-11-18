# Fiche modèle — AeroVision PatchCore

| Champ | Valeur |
|---|---|
| Tâche | Détection et segmentation d'anomalies visuelles, classification sain / défectueux |
| Méthode | PatchCore (Roth et al., CVPR 2022), implémentation propre au projet |
| Backbone | WideResNet-50-2 pré-entraîné ImageNet, couches 2 et 3, descripteurs réduits à 1 024 dims |
| Mémoire | coreset glouton k-center, 10 % des patchs (1 % pour l'entraînement « all ») |
| Entrée | image RGB redimensionnée 256×256 |
| Sorties | score image, carte d'anomalie à la résolution d'origine, régions, confiance calibrée |
| Données | MVTec AD (CC BY-NC-SA 4.0), un modèle par catégorie |
| Calibration | seuil F1 max + régression logistique 1D sur 30 % du test ; évaluation sur les 70 % restants |

## Résultats

À remplir après `python tasks.py train-all` : le script écrit `models/patchcore/RESULTS.md` et un rapport
JSON par catégorie. Mettre en avant, dans cet ordre :

1. **Taux de faux négatifs** au seuil calibré (défauts manqués) ;
2. AUROC image et AUROC pixel ;
3. taux de faux positifs (charge de revue pour l'opérateur) ;
4. temps d'inférence CPU / GPU.

Ordre de grandeur publié pour PatchCore sur MVTec AD : AUROC image proche de 0,99 en moyenne.
Ne pas recopier ce chiffre : publier ceux obtenus avec ce code et ce protocole.

## Limites connues

- Sensible aux variations d'acquisition absentes de l'entraînement (éclairage, pose, optique).
- Images redimensionnées à 256 px : les défauts de quelques pixels sur l'image d'origine
  peuvent disparaître. Piste : tuilage de l'image haute résolution.
- Le score image est le maximum des patchs : une seule zone bruitée peut déclencher une alerte.
- Calibration estimée sur peu d'exemples (quelques dizaines par catégorie) : la confiance
  affichée a une incertitude importante.
- Aucune validation sur pièce aéronautique réelle (voir cadre-aeronautique.md).
