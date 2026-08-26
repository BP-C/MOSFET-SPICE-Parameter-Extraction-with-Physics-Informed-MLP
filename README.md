README — MOSFET SPICE Parameter Extraction with Physics-Informed MLP


Ce projet utilise des réseaux de neurones pour prédire des paramètres
SPICE de MOSFET à partir de courbes électriques simulées.

Il contient deux notebooks principaux :

    1. mosfet_surrogate.ipynb
    2. mosfet_mlp.ipynb

Le premier entraîne un modèle direct (« surrogate ») qui reconstruit les
courbes électriques à partir des paramètres SPICE.

Le second entraîne un MLP inverse qui prédit les paramètres SPICE à
partir des courbes électriques, avec des contraintes physiques sur Vth,
Idlin, Idsat et Gmmax.


1. DATASET


Cette version du projet utilise un dataset de 3 000 échantillons.

Ce dataset est une portion du dataset initial de 50 000 échantillons.
Chaque échantillon correspond à un MOSFET simulé avec :

    - 101 points de tension de grille Vg ;
    - 12 canaux de courbes électriques ;
    - 11 paramètres SPICE associés ;
    - 4 valeurs de tension de seuil Vth ;
    - 7 caractéristiques Idlin, Idsat et Gmmax.

Les données sont réparties de manière reproductible avec SEED = 42 :

    - 80 % entraînement : 2 400 échantillons ;
    - 10 % validation   :   300 échantillons ;
    - 10 % test         :   300 échantillons.


2. INSTALLATION


Créer de préférence un environnement virtuel Python.

Sous Windows :

    python -m venv mosfet_env
    mosfet_env\Scripts\activate

Sous Linux ou macOS :

    python3 -m venv mosfet_env
    source mosfet_env/bin/activate

Installer les dépendances avec :

    pip install -r requirements.txt

Les bibliothèques principales sont NumPy, Pandas, Scikit-learn,
Matplotlib, Joblib et PyTorch.

Le projet utilise CUDA automatiquement si un GPU NVIDIA compatible est
disponible. Sinon, les notebooks sont exécutés sur CPU. [2] [4]


3. FICHIERS DE DONNÉES REQUIS


Les fichiers de données doivent être placés dans le dossier :

    dataset_3000/

Fichiers requis :

    dataset_3000/mosfet_X_dataset_3000.npz
        Contient les courbes électriques sous la clé "X".
        Forme attendue :

            (3000, 101, 12)

    dataset_3000/Y_3000.csv
        Contient les 11 paramètres SPICE de référence.
        Format : séparateur point-virgule (;), sans en-tête.

        Forme attendue :

            (3000, 11)

    dataset_3000/vth_extracted_3000.csv
        Contient les quatre valeurs de tension de seuil Vth pour les
        conditions Vbs = 0 V, -2 V, -6 V et -10 V.

        Forme attendue :

            (3000, 4)

    dataset_3000/currents_and_gmmax_extracted_3000.csv
        Contient sept caractéristiques électriques :

            idlin_0vbs
            idlin_m2vbs
            idlin_m6vbs
            idlin_m10vbs
            idsat_0vbs
            idsat_m2vbs
            max_dId_dVg_0vbs

        Forme attendue :

            (3000, 7)


4. PARAMÈTRES SPICE PRÉDITS


Les 11 paramètres SPICE prédits sont :

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

Les paramètres u0, ua, ub et uc sont transformés avec :

    log10(abs(valeur))

avant normalisation. La transformation inverse est appliquée lors de
l'évaluation des prédictions. [2] [3]


5. COURBES ÉLECTRIQUES


Chaque échantillon contient 12 canaux électriques :

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

Les courbes sont normalisées canal par canal dans l'intervalle [0, 1].

Avant d'être fournies au MLP, elles sont aplaties dans l'ordre
« canal-major » :

    (N, 101, 12) -> (N, 12, 101) -> (N, 1212)

Chaque entrée du MLP contient donc 1 212 valeurs. [2] [3]


6. ÉTAPE 1 — ENTRAÎNEMENT DU SURROGATE


Ouvrir le notebook :

    mosfet_surrogate.ipynb

Exécuter les cellules dans l'ordre :

    1. Config
    2. Model
    3. Dataprep
    4. Train
    5. Test

Le surrogate apprend la relation directe :

    Paramètres SPICE normalisés (11)
        ->
    Courbes électriques normalisées (12 × 101 = 1212)

Son architecture est un réseau dense :

    11 -> 512 -> 1024 -> 2048 -> 2048 -> 1212

Le notebook crée le dossier :

    mosfet_surrogate_3000_10_10_25C/

Le meilleur modèle est sauvegardé sous :

    mosfet_surrogate_3000_10_10_25C/surrogate_best.pth

Ce fichier est indispensable pour entraîner et évaluer le MLP inverse.

La cellule de test calcule les métriques MAE, RMSE et R² globales, ainsi
que les erreurs par canal. Il est recommandé de vérifier les performances
du surrogate avant de passer à l'étape suivante. [3]


7. ÉTAPE 2 — ENTRAÎNEMENT DU MLP INVERSE


Ouvrir le notebook :

    mosfet_mlp.ipynb

Exécuter les cellules dans l'ordre :

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

Le modèle contient cinq couches cachées de 2 048 neurones avec BatchNorm,
ReLU et Dropout.

La sortie est contrainte par des bornes physiques appliquées dans l'espace
normalisé à l'aide d'une sigmoïde redimensionnée. [2]

Le notebook crée le dossier :

    mosfet_mlp_3000_10_10_25C/

Le meilleur modèle est sauvegardé sous :

    mosfet_mlp_3000_10_10_25C/mlp_best.pth


8. FONCTION DE COÛT


La loss totale du MLP est composée de trois termes :

    Loss totale =
        Loss paramètres
      + LAMBDA_VTH × Loss Vth
      + LAMBDA_CONDUCTANCE × Loss conductance

1. Loss paramètres

Une Huber Loss compare les paramètres SPICE prédits aux paramètres de
référence.

2. Loss Vth

Les paramètres prédits passent dans le surrogate gelé pour reconstruire
les courbes log(Id).

Une interpolation linéaire différentiable impose la condition :

    logId(Vth) = -7

pour les quatre conditions de polarisation Vbs.

3. Loss conductance

La loss conductance compare les caractéristiques suivantes :

    - Idlin à Vg = 5 V pour Vbs = 0, -2, -6 et -10 V ;
    - Idsat à Vg = 5 V pour Vbs = 0 et -2 V ;
    - Gmmax, correspondant au maximum de dId/dVg.

Les coefficients configurés sont :

    LAMBDA_VTH = 0.285
    LAMBDA_CONDUCTANCE = 0.285

Le surrogate est gelé : ses poids ne sont pas mis à jour durant
l'entraînement du MLP, mais le gradient traverse le surrogate afin
d'optimiser le MLP. [2]


9. ÉVALUATION DU MLP


La cellule predict.py effectue les opérations suivantes :

    - chargement de mlp_best.pth ;
    - prédiction des paramètres SPICE sur le jeu de test ;
    - dénormalisation des paramètres ;
    - reconstruction des courbes avec le surrogate ;
    - calcul de MAE, RMSE et erreurs relatives ;
    - extraction et comparaison de Vth ;
    - comparaison de Idlin, Idsat et Gmmax ;
    - génération de graphiques « prédiction vs vrai ».

Les résultats sont enregistrés dans :

    mosfet_mlp_3000_10_10_25C/

Fichiers générés :

    Y_pred_test_mlp.npy
    Y_true_test_mlp.npy

    Vth_pred_test_mlp.npy
    Vth_true_test_mlp.npy

    Conductance_pred_test_mlp.npy
    Conductance_true_test_mlp.npy

Figures générées :

    grouped_parameters_pred_vs_true.png
    grouped_vth_pred_vs_true.png
    grouped_currents_pred_vs_true.png


10. TEST D'UN ÉCHANTILLON INDIVIDUEL


La cellule predict_one_Xtest_sample.py permet de tester visuellement un
échantillon du jeu de test.

Modifier la variable suivante :

    sample_index = 100

L'index doit être compris entre 0 et 299.

La cellule affiche :

    - les paramètres SPICE prédits et réels ;
    - les erreurs absolues et relatives ;
    - les courbes de référence ;
    - les courbes reconstruites par le surrogate.

Les paramètres prédits sont sauvegardés dans :

    parameters_Xtest_sample_<index>_mlp.txt


11. UTILISATION RAPIDE


    1. Créer et activer un environnement virtuel Python.

    2. Installer les dépendances :

           pip install -r requirements.txt

    3. Vérifier que le dossier dataset_3000/ contient les quatre fichiers
       de données requis.

    4. Exécuter mosfet_surrogate.ipynb :

           Config -> Model -> Dataprep -> Train -> Test

    5. Vérifier la présence du fichier :

           mosfet_surrogate_3000_10_10_25C/surrogate_best.pth

    6. Exécuter mosfet_mlp.ipynb :

           config.py
           model_mlp.py
           data_prep.py
           surrogate_vth_loss.py
           train.py
           predict.py

    7. Consulter les résultats dans :

           mosfet_mlp_3000_10_10_25C/
