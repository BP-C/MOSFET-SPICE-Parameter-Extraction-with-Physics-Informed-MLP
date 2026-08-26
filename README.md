======================================================================
README — Projet MOSFET : Surrogate + MLP inverse
======================================================================

Ce projet permet de prédire des paramètres SPICE de MOSFET à partir de
courbes électriques simulées.

Il comporte deux notebooks principaux :

    1. mosfet_surrogate.ipynb
    2. mosfet_mlp.ipynb

Le premier notebook entraîne un modèle direct, appelé « surrogate », qui
reconstruit les courbes électriques à partir des paramètres SPICE.

Le second notebook entraîne un MLP inverse qui prédit les paramètres
SPICE à partir des courbes électriques.

======================================================================
1. INSTALLATION
======================================================================

Le projet contient un fichier requirements.txt regroupant les dépendances
Python nécessaires.

Créer de préférence un environnement virtuel.

Sous Windows :

    python -m venv mosfet_env
    mosfet_env\Scripts\activate

Sous Linux ou macOS :

    python3 -m venv mosfet_env
    source mosfet_env/bin/activate

Installer ensuite les dépendances :

    pip install -r requirements.txt

Les principales bibliothèques utilisées sont :

    - numpy
    - pandas
    - scikit-learn
    - matplotlib
    - joblib
    - torch

Le fichier requirements.txt inclut PyTorch avec support CUDA :

    torch==2.11.0+cu128

Une carte graphique NVIDIA et des pilotes compatibles sont recommandés
pour accélérer les entraînements. Les notebooks sélectionnent
automatiquement CUDA lorsqu'il est disponible ; sinon, ils fonctionnent
sur CPU. [2] [4]

======================================================================
2. FICHIERS DE DONNÉES REQUIS
======================================================================

Les fichiers suivants doivent être présents dans le dossier de travail
avant l'exécution des notebooks :

    mosfet_X_dataset_4.npz
        Contient les courbes électriques sous la clé "X".

        Forme attendue :

            (50000, 101, 12)

    Y.csv
        Contient les paramètres SPICE de référence.

        Format attendu :
            - séparateur : point-virgule (;)
            - sans en-tête
            - 11 colonnes

    vth_extracted.csv
        Contient quatre valeurs de tension de seuil Vth par échantillon,
        pour les conditions Vbs = 0 V, -2 V, -6 V et -10 V.

        Forme attendue :

            (50000, 4)

    currents_and_gmmax_extracted.csv
        Contient les sept caractéristiques électriques suivantes :

            idlin_0vbs
            idlin_m2vbs
            idlin_m6vbs
            idlin_m10vbs
            idsat_0vbs
            idsat_m2vbs
            max_dId_dVg_0vbs

        Forme attendue :

            (50000, 7)

======================================================================
3. PARAMÈTRES SPICE PRÉDITS
======================================================================

Les onze paramètres SPICE prédits sont :

    vth0
    k1
    k2
    nfactor
    u0
    ua
    ub
    a0
    ags
    uc
    keta

Les colonnes correspondant à u0, ua, ub et uc sont transformées avec :

    log10(abs(valeur))

avant la normalisation. La transformation inverse est appliquée lors de
l'évaluation des prédictions. [2] [3]

======================================================================
4. DONNÉES ÉLECTRIQUES
======================================================================

Chaque échantillon comprend 101 points de tension de grille Vg et
12 canaux électriques :

    0.  logId_Vds_0p1_Vbs_0
    1.  logId_Vds_0p1_Vbs_2
    2.  logId_Vds_0p1_Vbs_6
    3.  logId_Vds_0p1_Vbs_10
    4.  Id_Vds_0p1_Vbs_0
    5.  Id_Vds_0p1_Vbs_2
    6.  Id_Vds_0p1_Vbs_6
    7.  Id_Vds_0p1_Vbs_10
    8.  Id_Vds_5p0_Vbs_0
    9.  Id_Vds_5p0_Vbs_2
    10. dId_dVg_Vds_0p1_Vbs_0
    11. Vg

Les courbes sont normalisées canal par canal dans l'intervalle [0, 1]
avec MinMaxScaler.

Avant d'être données au réseau, elles sont aplaties dans l'ordre
« canal-major » :

    (N, 101, 12) -> (N, 12, 101) -> (N, 1212)

Le MLP reçoit donc un vecteur de 1212 valeurs par échantillon. [2] [3]

======================================================================
5. ÉTAPE 1 — ENTRAÎNEMENT DU SURROGATE
======================================================================

Ouvrir le notebook :

    mosfet_surrogate.ipynb

Exécuter les cellules dans l'ordre suivant :

    1. Config
    2. Model
    3. Dataprep
    4. Train
    5. Test

Le surrogate apprend la relation directe suivante :

    Paramètres SPICE normalisés (11)
        ->
    Courbes électriques normalisées (12 × 101 = 1212)

Son architecture est un réseau dense avec les couches suivantes :

    11 -> 512 -> 1024 -> 2048 -> 2048 -> 1212

Le dossier de sortie créé est :

    mosfet_surrogate_10_10_25C/

Les principaux fichiers générés sont :

    X_train_conv.npy
    X_val_conv.npy
    X_test_conv.npy

    X_train_flat.npy
    X_val_flat.npy
    X_test_flat.npy

    Y_train_norm.npy
    Y_val_norm.npy
    Y_test_norm.npy
    Y_test_processed.npy

    scaler_x_minmax.pkl
    scaler_y_minmax.pkl

    param_bounds_lower_norm.npy
    param_bounds_upper_norm.npy

    surrogate_best.pth

Le fichier suivant est indispensable avant de lancer le notebook MLP :

    mosfet_surrogate_10_10_25C/surrogate_best.pth

La cellule de test du surrogate calcule des métriques globales et par
canal, notamment MAE, RMSE et R². Il est recommandé de vérifier ces
résultats avant de poursuivre, car le MLP utilise ce modèle pour les
contraintes physiques. [3]

======================================================================
6. ÉTAPE 2 — ENTRAÎNEMENT DU MLP INVERSE
======================================================================

Ouvrir le notebook :

    mosfet_mlp.ipynb

Exécuter les cellules dans l'ordre suivant :

    1. config.py
    2. model_mlp.py
    3. data_prep.py
    4. surrogate_vth_loss.py
    5. train.py
    6. predict.py

Le MLP apprend la relation inverse :

    Courbes électriques normalisées (1212)
        ->
    Paramètres SPICE normalisés (11)

L'architecture du MLP comprend cinq couches cachées de 2048 neurones,
avec BatchNorm, activation ReLU et Dropout.

Les sorties sont limitées à des bornes physiques définies pour chaque
paramètre. Ces bornes sont appliquées dans l'espace normalisé au moyen
d'une sigmoïde redimensionnée. [2]

Le dossier de sortie créé est :

    mosfet_mlp_10_10_25C/

Le meilleur modèle est enregistré sous le nom :

    mosfet_mlp_10_10_25C/mlp_best.pth

======================================================================
7. FONCTION DE COÛT DU MLP
======================================================================

Le MLP est entraîné avec une fonction de coût composée de trois termes :

    Loss totale =
        Loss paramètres
      + LAMBDA_VTH × Loss Vth
      + LAMBDA_CONDUCTANCE × Loss conductance

1. Loss paramètres
------------------

Une Huber Loss compare les paramètres SPICE prédits aux paramètres de
référence, dans l'espace normalisé.

2. Loss Vth
-----------

Les paramètres prédits par le MLP sont envoyés dans le surrogate gelé,
qui reconstruit les courbes log(Id).

Une interpolation linéaire différentiable est utilisée afin d'imposer,
pour chaque condition de Vbs, la relation :

    logId(Vth) = -7

3. Loss conductance
-------------------

La loss conductance compare les caractéristiques électriques extraites
des courbes reconstruites :

    - Idlin à Vg = 5 V pour Vbs = 0, -2, -6 et -10 V ;
    - Idsat à Vg = 5 V pour Vbs = 0 et -2 V ;
    - Gmmax, correspondant au maximum de dId/dVg.

Les coefficients actuellement configurés sont :

    LAMBDA_VTH = 0.285
    LAMBDA_CONDUCTANCE = 0.285

Le surrogate est gelé pendant l'entraînement du MLP : ses poids ne sont
pas modifiés, mais le gradient peut traverser le surrogate afin
d'optimiser le MLP. [2]

======================================================================
8. PARAMÈTRES D'ENTRAÎNEMENT
======================================================================

Les principaux hyperparamètres du MLP sont :

    batch_size  = 512
    num_epochs  = 150
    patience    = 40
    learning rate = 3e-5

L'optimiseur utilisé est AdamW avec :

    weight_decay = 1e-5

Un scheduler ReduceLROnPlateau réduit le taux d'apprentissage lorsque la
loss de validation ne s'améliore pas. Un mécanisme d'early stopping arrête
l'entraînement après 40 époques sans amélioration de la loss de
validation. [2]

======================================================================
9. ÉVALUATION DU MLP
======================================================================

La cellule predict.py réalise les opérations suivantes :

    - charge le modèle mlp_best.pth ;
    - charge les données de test et les scalers ;
    - prédit les paramètres SPICE du jeu de test ;
    - dénormalise les paramètres prédits et les références ;
    - reconstruit les courbes électriques à l'aide du surrogate ;
    - calcule les erreurs MAE, RMSE et relatives par paramètre ;
    - extrait et compare les valeurs Vth ;
    - compare Idlin, Idsat et Gmmax ;
    - crée des graphiques « prédiction vs vrai ».

Les résultats enregistrés dans le dossier mosfet_mlp_10_10_25C/ sont :

    Y_pred_test_mlp.npy
    Y_true_test_mlp.npy

    Vth_pred_test_mlp.npy
    Vth_true_test_mlp.npy

    Conductance_pred_test_mlp.npy
    Conductance_true_test_mlp.npy

Les figures générées sont :

    grouped_parameters_pred_vs_true.png
    grouped_vth_pred_vs_true.png
    grouped_currents_pred_vs_true.png

L'extraction de Vth repose sur une interpolation linéaire au croisement :

    logId(Vth) = -7

Si une courbe ne croise pas cette valeur cible, la valeur Vth associée
est définie comme NaN. [2]

======================================================================
10. TEST D'UN ÉCHANTILLON DU JEU DE TEST
======================================================================

La cellule predict_one_Xtest_sample.py permet d'évaluer visuellement un
échantillon individuel du jeu de test.

Modifier la variable suivante pour sélectionner un échantillon :

    sample_index = 100

L'index doit être compris entre 0 et le nombre d'échantillons du jeu de
test moins un.

La cellule :

    - charge l'échantillon sélectionné ;
    - prédit ses paramètres SPICE avec le MLP ;
    - compare les paramètres prédits aux paramètres réels ;
    - calcule les erreurs absolues et relatives ;
    - reconstruit les courbes avec le surrogate ;
    - affiche les courbes de référence et les courbes reconstruites.

Les paramètres prédits sont sauvegardés dans le fichier :

    parameters_Xtest_sample_<index>_mlp.txt

======================================================================
11. UTILISATION RAPIDE
======================================================================

Première utilisation complète :

    1. Créer un environnement virtuel Python.
    2. Activer l'environnement.
    3. Installer les dépendances :

           pip install -r requirements.txt

    4. Placer les fichiers de données requis dans le dossier du projet.
    5. Exécuter mosfet_surrogate.ipynb :

           Config -> Model -> Dataprep -> Train -> Test

    6. Vérifier la présence du fichier :

           mosfet_surrogate_10_10_25C/surrogate_best.pth

    7. Exécuter mosfet_mlp.ipynb :

           config.py
           model_mlp.py
           data_prep.py
           surrogate_vth_loss.py
           train.py
           predict.py

    8. Consulter les métriques, fichiers de sortie et graphiques dans :

           mosfet_mlp_10_10_25C/

======================================================================
12. REMARQUES IMPORTANTES
======================================================================

- Le seed est fixé à 42 afin de produire un découpage
  train / validation / test reproductible.

- Les notebooks du surrogate et du MLP doivent utiliser le même jeu de
  données, le même ordre de paramètres et le même ordre de canaux.

- La classe MosfetSurrogate présente dans mosfet_mlp.ipynb doit rester
  identique à celle utilisée pour entraîner surrogate_best.pth.

- Le fichier surrogate_best.pth doit être disponible avant
  l'entraînement ou l'évaluation du MLP.

- Les courbes externes ou les données de test doivent respecter l'ordre
  de canaux et l'aplatissement canal-major utilisés lors de
  l'entraînement.

======================================================================
