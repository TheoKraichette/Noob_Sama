# Noob Sama - Gaming Agent

## Jeu choisi

Jeu : Eclipse Of Souls

Lien : https://kraich.itch.io/eclipse-of-souls

Type : roguelite action twin-stick jouable dans le navigateur via itch.io.

Raison du choix : le jeu tourne directement dans Chromium, expose un score de
run lisible rapidement et propose des objectifs simples a mesurer (survivre,
tuer des ennemis, terminer des salles). Il rentre donc dans la contrainte du
projet : obtenir un environnement jouable et une baseline en moins de 30
minutes, sans entrainement GPU lourd.

## Installation

Le projet utilise Python et Playwright pour piloter Chromium.

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m playwright install chromium
```

Verification rapide sans ouvrir de navigateur visible :

```bash
python scripts/check_env.py
```

## Validation initiale

Le jeu se lance correctement depuis la page itch.io avec le bouton `Run game`.
Une fois le canvas charge, le menu principal s'ouvre et une run peut etre lancee
avec `Espace`, puis validation du premier heros disponible avec `Espace`.

Etats reperes dans le code du jeu :

- `title` : menu principal.
- `heroSelect` : choix du heros.
- `play` : partie en cours.
- `gameover` : mort.
- `victory` : victoire.

Le jeu maintient aussi des statistiques internes de run :

- `kills` : ennemis tues.
- `rooms` : salles terminees.
- `floors` : etages termines.
- `time` : temps de survie / temps de run.

## Controles confirmes

Les entrees clavier sont basees sur les touches physiques.

- Deplacement : `WASD` cote code, affiche `ZQSD` sur clavier AZERTY.
- Tir : fleches directionnelles, ou `IJKL`.
- Dash : `Espace`.
- Competence du heros : `Shift`.
- Interaction : `E`.
- Inventaire : `Tab`.
- Pause : `Escape`.
- Relancer : `R`.

## Observations de l'agent

Pour la premiere version, l'agent n'utilise pas les pixels du canvas. Le wrapper
lit l'etat interne expose par le jeu :

- `state` : etat courant du jeu.
- `hp` et `max_hp` : vie du joueur.
- `x`, `y` : position du joueur dans la salle.
- `floor` : profondeur actuelle.
- `kills` : ennemis tues pendant la run.
- `rooms` : salles terminees pendant la run.
- `floors` : etages termines pendant la run.
- `time` : temps ecoule dans la run.
- `gold` : or disponible.
- `item_count` : nombre d'objets deja obtenus.
- `room_type` : type de salle (`start`, `combat`, `shop`, `treasure`, etc.).
- `room_cleared` : indique si la salle courante est consideree terminee.
- `abyss_gate` : indique la presence d'une porte abyss optionnelle.
- `doors_open` : indique si les portes de la salle sont ouvertes.
- `portal_active` : indique si le portail d'etage est actif.
- `enemy_count` : nombre d'ennemis actifs.
- `nearest_enemy` : ennemi actif le plus proche, avec position, HP, type et distance.
- `nearest_pickup` : pickup le plus proche, avec type, rarete d'objet et distance.
- `nearest_shop_item` : objet ou soin achetable le plus proche, avec prix.
- `nearest_door` : porte ouverte la plus proche.
- `near_choice` : objet de salle item actuellement selectionnable avec `E`.
- `portal` : position et distance du portail actif apres boss.
- `available_doors` : portes ouvertes disponibles, triees par distance.
- `pickups` : liste courte des pickups visibles.

Ce choix evite un apprentissage depuis pixels bruts, trop couteux pour deux
jours, et permet de mesurer rapidement si l'agent fait mieux que le hasard.

## Actions de l'agent

Les actions disponibles dans le wrapper sont volontairement simples :

- `noop`
- `up`
- `down`
- `left`
- `right`
- `shoot_up`
- `shoot_down`
- `shoot_left`
- `shoot_right`
- `dash`
- `up_shoot_up`
- `down_shoot_down`
- `left_shoot_left`
- `right_shoot_right`
- `interact`
- deplacements diagonaux
- combinaisons deplacement + tir independant, par exemple `left_shoot_right`
- `skill` (competence du heros) et ses variantes visees
- combinaisons **deplacement diagonal + tir**, par exemple
  `down_left_shoot_up_right`

Soit 96 actions au total. Les 32 dernieres ont ete ajoutees en phase 4 :
`js/core/input.js` resout le deplacement et la visee sur deux axes
independants, donc maintenir W + A + Fleche-haut est une entree legale que
l'espace d'actions n'offrait simplement pas. Sans elles, une direction de
fuite diagonale coutait le tir, et la politique s'immobilisait pour tirer une
fois sur deux (voir `docs/carnet_essais.md`, V35).

La baseline aleatoire conserve l'ancien espace d'actions simple pour rester
reproductible. Les actions enrichies servent surtout a l'agent entraine et
a la politique heuristique.

Une politique heuristique de reference est disponible :

```bash
.venv/bin/python scripts/play_heuristic.py --watch --episodes 1 --max-steps 120
```

Elle n'est pas l'agent entraine. Elle sert de guide de comportement : viser
l'ennemi proche, garder ses distances, ramasser les pickups utiles, interagir
avec les objets/shops accessibles, puis aller vers les portes ou portails.

Cas speciaux deja pris en compte par cette heuristique :

- item room / treasure room : aller vers l'objet propose et valider avec `E`
  quand le choix est a portee.
- shop : acheter seulement si l'or suffit et si l'achat est utile, par exemple
  soin uniquement quand il manque de la vie.
- boss room : combattre comme une salle normale, puis aller au portail central
  et valider avec `E` quand le boss est vaincu.
- salle du parieur (`gambler`) : seule salle optionnelle encore refusee, payer
  de l'or pour un tirage au sort etant de la variance sans plancher. Les
  autres (`altar`, `defi`, `scelle`) ont ete rouvertes en phase 3 : le butin
  pose sur un socle ne coute plus rien une fois le combat gagne.
- abyss gate : detectee, mais non priorisee pour la premiere version car c'est
  un risque optionnel.

## Recompense et score proposes

Pour commencer simple, le score principal sera base sur la progression mesurable
de la run :

```text
score = floors * 100 + rooms * 10 + kills + time * 0.1
```

Ce score garde un signal meme si l'agent meurt vite : survivre un peu rapporte,
nettoyer une salle rapporte davantage, et changer d'etage rapporte beaucoup.

La recompense utilisee par `step(action)` mesure la difference de score entre
deux observations et ajoute une penalite si le joueur perd de la vie :

```text
reward = delta_score - hp_lost * 2
```

La baseline demandee par la consigne sera mesuree avec l'agent aleatoire sur le
meme score et le meme nombre de parties que l'agent entraine.

## Baseline aleatoire

Le wrapper a ete accelere (une seule lecture d'etat par step au lieu de trois,
soit environ 4 steps par seconde au lieu de 2). Comme le jeu tourne en temps
reel, ce changement modifie le temps de jeu couvert par un meme nombre de
steps : la baseline a donc ete re-mesuree sur l'environnement accelere pour
garder une comparaison honnete. Autre effet du changement : la somme des
rewards d'un episode correspond maintenant exactement au score final moins les
HP perdus (avant, ce qui se passait entre deux steps n'etait compte nulle part).

Baseline re-mesuree le 20 aout 2026 avec :

```bash
.venv/bin/python scripts/baseline_random.py --episodes 20 --max-steps 100 --seed 42
```

Resultats :

- Episodes : 20.
- Budget : 100 steps maximum par episode.
- Seed : 42.
- Score moyen : 1.836.
- Score minimum : 1.742.
- Score maximum : 2.252.
- Ecart-type : 0.132.
- Reward moyenne : 1.799.
- Kills : 0 sur tous les episodes.
- Salles terminees : 0 sur tous les episodes.

Conclusion : l'agent aleatoire survit quelques secondes mais ne progresse pas
dans le donjon avec ce budget. Le premier agent entraine devra etre compare sur
le meme nombre d'episodes et le meme budget de steps.

Donnees : `reports/random_baseline.csv`.

## Courbe de progression

La courbe de progression est generee depuis les CSV de `reports/` sans
dependance graphique externe :

```bash
python scripts/plot_progress.py
```

Sortie : `reports/progression_curve.svg`.

La courbe affiche la moyenne mobile des scores d'entrainement, les evaluations
periodiques, le meilleur score d'evaluation conserve et la baseline aleatoire.
Elle sert de support visuel pour comparer l'agent aleatoire et l'agent entraine
pendant la restitution.

Elle trace une progression **par episode**, ce qui n'a de sens que pour les
methodes qui apprennent en jouant : le Q-learning de la phase 1 et la passe RL
de la phase 3. Le clonage des phases 3 et 4 apprend par epoques sur un jeu de
donnees fixe, et sa progression se lit dans les tableaux de resultats, pas sur
cette courbe.

## Methode d'apprentissage

Algo retenu : **Q-learning tabulaire sur etat discretise**. Pourquoi celui-la
pour ce jeu-la : l'environnement tourne en temps reel dans un navigateur
(~4-5 steps par seconde, pas de GPU), donc le budget total d'apprentissage est
d'environ 30 000 steps - le deep RL est hors budget d'echantillons. Les
observations structurees se discretisent naturellement, et une Q-table se
sauvegarde en JSON de quelques dizaines de Ko, rechargeable trivialement et
inspectable a l'oeil nu.

L'etat compresse chaque observation en une cle compacte (~120 situations
reellement rencontrees) : direction de l'ennemi le plus proche (8 secteurs),
sa distance (proche/moyen/loin, seuils 170/430 repris de l'heuristique), vie
basse ou non, direction relative en 8 secteurs de la porte cible quand la
salle est vide, pickup utile a portee, interaction possible. La porte cible
n'est pas la plus proche mais celle qui mene a la salle non visitee la plus
proche (petit parcours BFS du graphe du donjon expose par le jeu), sinon
l'agent fait des allers-retours entre salles deja nettoyees.

Quatre mecanismes completent le Q-learning de base :

- **Exploration guidee** : pendant l'exploration, 60 % des coups sont joues par
  l'heuristique (`src/policies.py`) au lieu du pur hasard. Le Q-learning etant
  off-policy, il apprend de ces demonstrations.
- **Meta-action de delegation** : une 16e action "heuristic" permet a l'agent
  de deleguer un coup a l'heuristique ; la Q-table apprend OU deleguer et ou
  jouer mieux qu'elle. Le plancher de performance devient l'heuristique, le
  plafond la depasse localement. En evaluation, seule la Q-table decide (y
  compris de deleguer) - c'est bien une politique apprise.
- **Shaping de porte** (entrainement seulement) : se rapprocher de la sortie
  d'une salle vide donne un petit signal continu (~+0.15/step) qui mene au
  +10/salle, trop rare pour etre decouvert seul. Le score de comparaison ne
  change pas.
- **Sauvegarde du meilleur** : eval greedy (epsilon=0) de 3 episodes tous les
  25 episodes ; si la moyenne bat le record, l'agent est sauvegarde dans
  `models/best_agent.json`. Un detecteur de blocage (8 steps sans bouger hors
  combat -> un coup aleatoire) joue en eval et en demo le role que
  l'exploration joue pendant l'entrainement.

Hyperparametres calcules depuis le budget d'echantillons : alpha 0.25
decroissant vers 0.05 en fin de run (fige les acquis, supprime l'oscillation
post-pic), gamma 0.97 (horizon ~33 steps, un trajet vers une porte), epsilon
1.0 -> 0.10 atteint a 70 % du run, penalite HP 2.0.

Commande du run complet (400 episodes, puis prolongations par reprise) :

```bash
python scripts/train.py --episodes 400 --max-steps 100 --seed 42 --gamma 0.97 --epsilon-min 0.1 --guide-ratio 0.6 --door-shaping 0.005
python scripts/train.py --resume --episodes 1200 --max-steps 100 --seed 42 --gamma 0.97 --epsilon-min 0.1 --guide-ratio 0.6 --door-shaping 0.005
```

## Resultats de l'entrainement

1 200 episodes x 100 steps au total (~6 h, en runs chaines par `--resume`),
scores dans `reports/training_scores.csv` (une ligne par episode, train et
eval, 48 points d'evaluation).

- Meilleur agent (eval@975) : score moyen **15.757**, soit **8.6x la baseline
  aleatoire (1.836)** - dans la fourchette de l'heuristique de reference.
- Progression du record au fil des evals : 6.74 (ep. 25) -> 10.64 (ep. 100)
  -> 11.66 (ep. 250) -> 12.05 (ep. 450) -> 14.98 (ep. 575) -> 15.20 (ep. 750)
  -> **15.76 (ep. 975)**. Sur la seconde moitie du run, la majorite des evals
  depassent 10.
- Pendant l'entrainement : 3 331 kills et 667 salles terminees en 1 200
  episodes (l'aleatoire : 0 kill, 0 salle en 20 parties).
- En demo longue (300 steps par partie), l'agent tourne autour de 15-30 points
  avec 1-2 salles nettoyees et 3-8 kills par partie.
- Le meilleur agent est conserve dans `models/best_agent.json` (128 etats,
  16 actions, ~60 Ko), rechargement verifie depuis un script neuf.

Mise a jour finale apres observation visuelle et ameliorations du controle bas
niveau :

- Meilleur checkpoint conserve : `models/best_agent.json`. Un record ponctuel
  `eval@1305` a **18.663** a ete observe pendant les essais, mais il ne doit
  pas etre lu comme la performance moyenne stable.
- Evaluation finale robuste sur 20 episodes x 100 steps :
  score moyen **8.973**, minimum **2.403**, maximum **15.379**
  (`reports/final_eval.csv`).
- Par rapport a la baseline aleatoire **1.836**, ce score moyen final reste
  environ **4.9x meilleur**.
- Les runs reussis terminent generalement 1 salle avec 3-4 kills ; les runs
  faibles viennent surtout de combats ou l'agent prend trop de degats.
- Les tirs en mouvement, tirs diagonaux et dashs directionnels ont ete ajoutes
  au controle bas niveau. Les apprendre directement dans la Q-table agrandit
  trop l'espace d'actions ; la meilleure version pratique reste donc le modele
  Q-learning sauvegarde avec garde-fous heuristiques.

## Phase 2 - environnement turbo et agent par demonstrations

La deuxieme phase du projet (details et echecs mesures dans
`docs/carnet_essais.md`, versions V9 a V15) a change d'echelle :

- **Environnement turbo** : le jeu simule a pas fixe 120 Hz ; le wrapper pilote
  cette boucle en synchrone (`EclipseEnv(turbo=True)`) au lieu de tenir les
  touches en temps reel. Vitesse mesuree : ~540 steps/s au lieu de 4-5, temps
  de jeu exact par step, mode temps reel conserve pour les demos.
- **Budget re-echantillonne** : episodes de 900 steps (le budget minimal pour
  atteindre le boss d'un etage), baseline aleatoire re-mesuree sur ce budget :
  **10.54** en moyenne sur 20 episodes (`reports/random_baseline_900.csv`).
- **DQN teste honnetement** : un Double DQN numpy (70 features continues,
  59 actions, replay, target network) a ete pousse sur ~800 episodes en
  5 iterations correctives. Resultat negatif documente : 6.66 en eval pure,
  sous la baseline aleatoire - le RL pur n'apprend pas le combat fin avec ce
  budget d'echantillons (`reports/final_eval_dqn.csv`).
- **Esquives dans l'heuristique** : le wrapper expose desormais les
  projectiles ennemis, les zones d'explosion telegraphees et l'attaque en
  cours des boss. L'heuristique esquive les orbes (pas de cote
  perpendiculaire), sort des cercles annonces, strafe les assauts re-vises
  du gardien et frappe pendant sa fenetre d'etourdissement. Chaque ajout a
  ete mesure par un audit des causes de mort (16 -> 9 morts sur 20
  episodes, pertes de vie face au gardien divisees par deux).
- **Agent retenu : clonage de comportement + DAgger**
  (`models/bc_agent.json`). 150 episodes de demonstrations heuristiques
  (117 151 paires, 100 features), puis une ronde DAgger : le clone joue
  100 episodes pendant que l'heuristique etiquette chacun de ses etats
  (69 638 paires) — le reseau apprend ainsi a se rattraper dans ses
  propres derives. Entropie croisee, 256x256, precision de validation
  85.6 %.

- **Conscience de la carte et du temps** : le graphe du donjon est route par
  priorite (tresor et marchand d'abord pour s'equiper, le gardien en
  dernier), les deplacements hors combat utilisent le dash, et l'agent se
  pre-positionne pendant la fenetre d'apparition des ennemis au lieu de
  figer (un defaut du professeur que le clone imitait fidelement).

Resultats du protocole final (20 episodes x 900 steps, meme score) :

| Agent | Score moyen | vs aleatoire | Salles | Etages | Morts |
| --- | --- | --- | --- | --- | --- |
| Aleatoire | 10.54 | 1x | 1 | 0 | 0/20 (survie passive) |
| DQN pur (meilleur checkpoint) | 6.66 | 0.6x | 0 | 0 | 20/20 |
| Heuristique (professeur) | 90.08 | 8.5x | 97 | 2 | 6/20 |
| **Reseau clone + 2 rondes DAgger (retenu)** | **93.23** | **8.8x** | 100 | **2** | 9/20 |

Le reseau retenu joue seul (aucun garde-fou, aucune delegation a
l'execution) : il nettoie 3 a 6 salles par episode, esquive projectiles et
zones d'explosion, route sa progression par la carte, dash entre les
objectifs et termine des etages (pire episode du protocole : 51 points).
Apres deux rondes DAgger, il egale son professeur - le plafond naturel de
l'imitation. L'affinage RL au-dela de ce niveau a ete tente
(`scripts/train_dqfd.py`, perte TD + ancre sur les demonstrations) et ne
l'a pas depasse dans le budget disponible : essai negatif documente au
carnet (V21). Donnees : `reports/final_eval_bc.csv`,
`reports/final_eval_heuristic.csv`.

Reproduction complete :

```bash
python scripts/collect_demos.py --episodes 150 --max-steps 900 --seed 42
python scripts/train_bc.py --hidden 256 256 --epochs 60
python scripts/collect_demos.py --policy models/bc_agent.json --episodes 100 --seed 7 --output data/dagger1.npz
python scripts/train_bc.py --demos data/demos.npz data/dagger1.npz --hidden 256 256 --epochs 60
python scripts/watch_dqn.py --agent models/bc_agent.json --episodes 3
```

Suivi d'un entrainement en direct (scores, salles, kills, etages) :

```bash
python scripts/dashboard.py
# puis ouvrir http://127.0.0.1:8765
```

## Phase 3 - viser la run complete

### Pourquoi le budget change

La phase 2 comparait sur 900 steps. La lecture du code du jeu donne la taille
reelle d'une partie : `FINAL_FLOOR = 5`, des etages de 8/9/10/11/8 salles,
soit 46 salles et 5 gardiens, et 15 a 19 minutes de jeu annoncees par les
auteurs. A 0.13 s de jeu par step, une run complete coute environ **9 000
steps** : sur 900, la victoire etait impossible par construction, pas par
faiblesse de l'agent.

Le protocole passe donc a **20 episodes x 9 000 steps**, avec la baseline
aleatoire re-mesuree sur ce budget :

```bash
python scripts/baseline_random.py --episodes 20 --max-steps 9000 --seed 42 --turbo
```

- Score moyen : **34.114** (min 2.129, max 134.951, ecart-type 32.827).
- 19 episodes sur 20 se terminent par une mort.
- Donnees : `reports/random_baseline_9000.csv`.

Les chiffres des budgets 100 et 900 steps restent au carnet d'essais comme
historique, mais ne sont plus le referentiel de comparaison.

### Ce qui est autorise, ce qui ne l'est pas

L'agent ne gagne aucun avantage que le jeu ne donne pas a un joueur.

**Autorise** : lire l'etat interne du jeu (choix documente depuis la phase 1,
pour eviter l'apprentissage depuis pixels) ; n'agir que par evenements
clavier ; avancer la simulation a pas fixe (mode turbo) ; calculer sans
limite de temps entre deux pas ; et demarrer un episode **d'entrainement** a
un etage arbitraire.

**Interdit** : modifier les regles ou la difficulte du jeu, octroyer vie, or
ou objets en evaluation, desactiver des ennemis, et conserver la moindre
meta-progression d'un episode a l'autre.

Deux garde-fous rendent ces regles verifiables plutot que declaratives :

- La sauvegarde du jeu (`localStorage` : heros debloques, eclats du
  Sanctuaire) est **videe a chaque chargement de page**. Sans cela elle
  s'accumulait entre les episodes d'une meme session, et les derniers
  tournaient sur un personnage plus fort que les premiers - et que la
  baseline. Consequence assumee : seul Solen, le heros de depart, est
  jouable.
- L'evaluation refuse de demarrer ailleurs qu'a l'etage 1 avec un inventaire
  vide (assertion dans `scripts/watch_dqn.py`). Le curriculum par etage
  n'existe donc que pour la collecte de donnees.

### L'audit des morts

Toutes les decisions de la phase 3 viennent d'un meme outil :

```bash
python scripts/audit_deaths.py --policy heuristic --episodes 20 --max-steps 9000
```

Il produit une ligne par point de vie perdu, avec le coupable nomme par le jeu
lui-meme (`killerName`), l'etage, le type de salle et l'attaque de gardien en
cours ; plus deux metriques qui manquaient : les **degats subis par salle
nettoyee** (progresser plus loin fait mecaniquement encaisser plus, seul le
taux dit si la politique s'ameliore) et les **blocages**, ces episodes qui
epuisent leur budget sans mourir et qu'aucune table de morts ne montre.

Trois trouvailles majeures en sont sorties, detaillees au carnet d'essais :

1. L'agent ne voyait qu'un tiers du danger (observations tronquees a 6
   projectiles et 6 zones, quand `aftershock` en pose 19 d'un coup).
2. Les salles de combat profondes contiennent des **piliers** qui arretent
   les tirs. Rien ne les exposait : l'agent tirait dedans, jusqu'a 7 380 steps
   d'affilee - 82 % d'un budget d'episode.
3. La politique combattait les elites des salles scellees puis **repartait
   sans la relique**, parce qu'une regle de prudence confondait « declencher
   une epreuve » et « ramasser un butin deja pose ».

### L'agent appris

Le pipeline part du professeur et n'en garde que ce qu'un reseau peut
reproduire :

1. **Collecte** : 180 episodes du professeur, dont un tiers demarres en
   profondeur par le curriculum -> 243 751 paires (features -> action),
   stockees en float16, avec sous-echantillonnage des couloirs calmes pour
   rendre le jeu de donnees majoritairement combattant.
2. **Clonage** : reseau 384x384 en numpy, entropie croisee.
3. **DAgger, deux rondes** : le clone joue, l'heuristique etiquette chacun de
   ses etats. C'est ce qui lui apprend a se rattraper dans ses propres
   derives - le defaut central du clonage pur.
4. **Auto-imitation iteree** et **affinage RL (DQfD)** : tentes, mesures,
   non retenus (voir le carnet).

Total : **581 438 paires**, precision de validation 83.7 %.

### Resultats

Protocole : 20 episodes x 9 000 steps, meme score, meme budget. Les colonnes
salles et etages sont des totaux sur les 20 episodes - elles accompagnent le
score parce qu'elles, au moins, ne se gagnent pas en restant immobile.

| Agent | Score moyen | vs aleatoire | Salles | Etages | Etage max | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| Aleatoire | 34.11 | 1x | 0 | 0 | 1 | 0/20 |
| Clone (imitation seule) | 186.95 | 5.5x | 117 | 9 | 2 | 0/20 |
| Clone + 1 ronde DAgger | 159.02 | 4.7x | 127 | 8 | 2 | 0/20 |
| Auto-imitation iteree | 158.36 | 4.6x | 131 | 8 | 2 | 0/20 |
| Affinage RL (DQfD) | 198.04 | 5.8x | 97 | 5 | 2 | 0/20 |
| **Clone + 2 rondes DAgger (retenu)** | **181.42** | **5.3x** | **137** | **10** | **3** | 0/20 |
| Heuristique (professeur) | 272.06 | 8.0x | 194 | 19 | **5** | **1/20** |

Le modele retenu n'est pas celui qui marque le plus. Deux lignes marquent plus
que lui et jouent moins bien :

- le clone sans DAgger doit ses 186.95 a **cinq episodes figes** sur 9 000
  steps, ou le terme `time * 0.1` du score paie environ 135 points de pure
  survie - davantage qu'une run honnete qui nettoie cinq salles et meurt ;
- l'agent RL, lui, a appris a survivre sans jouer : 13 episodes sur 20
  epuisent leur budget, pour 97 salles nettoyees contre 137.

Le score n'a pas ete modifie - c'est le referentiel de comparaison avec la
baseline. Mais il ne suffit pas seul a departager deux politiques, et le
carnet d'essais detaille les trois fois ou cette pathologie est apparue.

**Le professeur reste devant le clone** (272.06 contre 181.42), contrairement
a la phase 2 ou le clone l'avait rattrape. Deux raisons mesurables : l'espace
d'actions est passe de 59 a 64 et l'etat de 109 a 202 dimensions ; et le
professeur tire desormais une part de sa force de decisions **rares** -
prendre l'autel, classer trois reliques, garder l'or pour un coeur - qui
n'apparaissent que quelques dizaines de fois dans 580 000 paires.

La run victorieuse du professeur a traverse les 5 etages, nettoye 32 salles,
vaincu 227 ennemis et accumule 13 reliques. Les runs qui meurent a l'etage 1
en ont une seule : le build n'est pas un confort, c'est la variable qui
decide.

### Reproduction

```bash
# Baseline du protocole
python scripts/baseline_random.py --episodes 20 --max-steps 9000 --seed 42 --turbo

# Ou meurt la politique, et de quoi
python scripts/audit_deaths.py --policy heuristic --episodes 20 --max-steps 9000

# Collecte (curriculum : 60 % de runs completes, le reste en profondeur)
python scripts/collect_demos.py --episodes 180 --max-steps 9000 --workers 6 \
    --floors 1 1 1 1 1 1 2 3 4 5 --seed 42 --output data/demos.npz
python scripts/train_bc.py --demos data/demos.npz --hidden 384 384 --epochs 40 \
    --batch-size 512 --output models/bc_agent_v3.json

# Ronde DAgger : le clone joue, l'heuristique etiquette
python scripts/collect_demos.py --policy models/bc_agent_v3.json --episodes 120 \
    --max-steps 9000 --workers 6 --floors 1 1 1 1 2 3 --seed 7 --output data/dagger1.npz
# ... puis une deuxieme ronde, et l'entrainement final :
python scripts/train_bc.py --demos data/demos.npz data/dagger1.npz data/dagger2.npz \
    --hidden 384 384 --epochs 40 --batch-size 512 --output models/bc_agent_v5.json

# Evaluation au protocole
python scripts/watch_dqn.py --agent models/bc_agent_v5.json --episodes 20 \
    --max-steps 9000 --headless --turbo --output reports/eval_bc_v5.csv
```

## Phase 4 - la ou la run perdait vraiment ses coeurs

La phase 3 avait cherche la profondeur ; la phase 4 est partie d'un chiffre
tres simple. Le joueur a **6 demi-coeurs** (`PLAYER_CFG.maxHp`), la politique
en perdait **1.0 par salle**, et une run complete fait 46 salles. Rien
d'autre ne compte tant que ce rapport n'a pas baisse.

L'audit des morts a designe une cause qui n'etait ni la perception ni
l'esquive : **171 des 261 coups recus l'etaient sans un seul projectile a
l'ecran**, et 92 avec un seul ennemi present. La politique se faisait toucher
au corps a corps par des ennemis deux fois plus lents qu'elle.

### Deux trous structurels

**1. L'espace d'actions etait incomplet.** `js/core/input.js` resout le
deplacement (`getMove`) et la visee (`getShoot`) sur deux axes independants :
maintenir W + A + Fleche-haut est une entree parfaitement legale. Or aucune
action « deplacement diagonal + tir » n'existait. Verifie dans le navigateur
avant d'y toucher : `vx=-205 vy=-205` (norme 290) avec `aim=[1,0]`.

Mesure sur 5 341 pas de combat reels : la politique ne tirait que pendant
**68,3 %** d'entre eux. Des que la fuite etait diagonale — une fois sur deux
— elle lachait le tir et s'immobilisait. 32 actions ajoutees, 96 en tout, et
le temps de tir passe a **92,6 %**. Ces actions representent aujourd'hui
21,8 % des decisions du professeur.

**2. La visee clavier n'a que huit rayons.** Une cible a un demi-secteur du
rayon est manquee de `r x sin(22.5 deg)`, soit 115 px a 300 px de distance,
contre un corps et un orbe qui font ensemble 20 px. Sur 4 948 tirs reels,
**seuls 25,6 % pouvaient geometriquement atteindre leur cible**.

La reponse n'est pas de mieux tirer, c'est de mieux se placer : quand la
cible n'est pas sur un rayon, **tourner autour d'elle** jusqu'a ce qu'elle y
soit, sans cesser de tirer. Plus le choix du rayon reellement le plus proche
du relevement, la regle precedente basculant en diagonale des que les deux
ecarts depassaient 45 px. Tirs pouvant toucher : **25,6 % -> 40,8 %**, ecart
lateral median 52 -> 31 px.

### Le protocole de mesure aussi a change

Vingt episodes laissent une erreur type d'environ 25 points sur la moyenne,
qui masque tout ce qui est plus petit qu'elle. `audit_deaths.py` accepte
desormais `--workers`, les essais se font sur **40 episodes**, et la
metrique de decision est les **PV perdus par salle** : elle compte des
centaines d'evenements au lieu de quarante, donc son erreur est bien plus
serree que celle du score.

Cette prudence vaut aussi contre nous. Re-mesuree sur 40 episodes, la
baseline aleatoire donne **17.13** au lieu des 34.11 mesures sur 20 : deux
episodes campes a 135 et 109 points portaient la moitie de l'ancienne
moyenne. Les rapports « x fois la baseline » ci-dessous continuent d'utiliser
**34.11**, le chiffre le plus haut, donc le plus defavorable a l'agent.

### Resultats

Protocole : **40 episodes x 9 000 steps**. Salles et etages sont des totaux
sur les 40 episodes.

| Agent | Score moyen | Salles | Etages | Etage max | PV / salle | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| Aleatoire | 17.13 +/- 3.3 | 2 | 0 | 1 | - | 0/40 |
| Clone + 2 rondes DAgger | 264.94 +/- 18.2 | 383 | 37 | 3 | 0.95 | 0/40 |
| Clone 512x512 | 260.59 +/- 17.7 | 362 | 41 | 3 | 0.90 | 0/40 |
| **Clone phase 4 (retenu, 80 ep.)** | **264.8 +/- 14.8** | **759** | **79** | **4** | **0.87** | 0/80 |
| Professeur phase 4 | **347.06 +/- 36.9** | 485 | 52 | **5** | **0.84** | **1/40** |

Le rapport a la baseline depend de laquelle on prend, et les deux sont
donnees : **15.5x** contre la baseline re-mesuree sur 40 episodes (17.13),
**7.8x** contre celle de la phase 3 (34.11, mesuree sur 20 episodes dont deux
campes). Le professeur est a 20.3x et 10.2x respectivement. La colonne
« PV / salle » ne s'applique pas a l'aleatoire, qui ne termine presque aucune
salle : 2 en 40 episodes.

Sur le protocole historique de 20 episodes, le modele retenu, recharge depuis
un script neuf, marque **242.42 +/- 20.3** pour 169 salles et 18 etages
franchis, contre 181.42 / 137 salles / 10 etages pour le champion de la
phase 3 : **+34 % de score, +23 % de salles, +80 % d'etages**, et surtout
**aucun episode fige** sur les 20 (contre 2 auparavant), donc pas un point
gagne en restant immobile.

### Ce qui n'a pas marche

Quatre essais mesures et non retenus, detailles au carnet :

- **Anticiper le mouvement de la cible** (viser le point d'impact plutot que
  la position courante). La visee s'ameliore, le jeu se degrade : les
  poursuivants accelerent a 850 px/s2 et oscillent, leur vitesse instantanee
  predit mal une demi-seconde, et comme le point vise pilote aussi le
  deplacement, une prediction bruitee fait tourner la politique dans le vide.
- **Aller chercher le soin de plus loin** en combat. Traverser une salle pour
  un demi-coeur coute plus de vie qu'il n'en rend.
- **Les rondes DAgger**, et **un reseau plus large**. La precision
  d'imitation monte de 81.7 a 84.1 %, le jeu ne bouge pas : les quatre
  clones tiennent dans 25 points pour des erreurs types de 18 a 22. C'est le
  resultat le plus net de cette phase cote apprentissage — l'imitation sature
  bien avant son professeur, et ce n'est pas la capacite qui manque.

### Ce qui reste

Le clone plafonne a 76 % du professeur. Le professeur, lui, franchit les
cinq etages et gagne une run sur quarante, mais perd encore 0.84 PV par
salle quand il en faudrait environ 0.2 pour terminer regulierement les 46
salles avec six demi-coeurs et un soin rare. L'ecart n'est plus un probleme
de perception : c'est le cout de chaque salle qui doit encore etre divise
par quatre.

### Reproduction

```bash
# Baseline du protocole (40 episodes)
python scripts/baseline_random.py --episodes 40 --max-steps 9000 --seed 42 --turbo \
    --output reports/random_baseline_9000_40.csv

# Le professeur, et ou il perd ses coeurs
python scripts/audit_deaths.py --policy heuristic --episodes 40 --workers 4 \
    --output reports/death_audit_v39_ray.csv

# Collecte : 150 episodes avec curriculum, puis 200 depuis l'etage 1
python scripts/collect_demos.py --episodes 150 --workers 6 --floors 1 1 1 1 2 3 4 5 \
    --output data/demos_v4.npz
python scripts/collect_demos.py --episodes 200 --workers 6 --floors 1 --seed 777 \
    --output data/demos_v4b.npz

# Clonage
python scripts/train_bc.py --demos data/demos_v4.npz data/demos_v4b.npz \
    --hidden 384 384 --epochs 30 --output models/bc_agent_v6.json

# Evaluation : 40 episodes pour decider, 20 pour le protocole historique
python scripts/audit_deaths.py --policy models/bc_agent_v6.json --episodes 40 \
    --workers 3 --output reports/death_audit_clone_v6.csv
python scripts/watch_dqn.py --agent models/bc_agent_v6.json --episodes 20 \
    --max-steps 9000 --headless --turbo --output reports/eval_bc_v6.csv
```

## Phase 5 - viser juste, esquiver vraiment, finir la run

La phase 4 s'arretait sur un constat : 0.84 PV perdu par salle quand il en
faudrait environ 0.2 pour finir les 46 salles avec six demi-coeurs. L'audit
disait ou : **47 % des PV au corps a corps pur** (aucun projectile, aucune
zone a l'ecran), 39 % avec un seul ennemi present, et le deuxieme tueur du
jeu etait l'Ame errante, l'ennemi le plus lent du bestiaire. La cause etait
la visee : huit rayons clavier, 41 % des tirs capables de toucher, et des
regles qui font marcher le heros autour de sa cible pour l'aligner - donc
vers elle.

### Trois changements de methode

**Lire les regles a la source.** Le jeu est public et son code aussi
(`js/data/enemies.js`, `js/game/enemy.js`, `js/game/boss.js`,
`js/data/items.js`, `js/data/difficulty.js`, version 4.1.0, celle que sert
itch). Vitesse et rayon de chaque famille, duree exacte de chaque
telegraphe (un chargeur verrouille sa visee a 0,28 s et part a 0,5 s pour
0,34 s ; un tireur tire a 0,52 s ; une bombe explose apres sa meche sur son
rayon de souffle), attaques nommees des onze gardiens, effets des 159
reliques : plus une constante n'est devinee (`src/bestiary.py`). Le code
sert a lire les regles ; l'agent joue toujours la version itch.

**Viser a la souris.** `js/core/input.js` offre une visee a 360 degres au
clic gauche, entree officielle de tout joueur ; les fleches gardent la
priorite. L'environnement la pilote (`step(action, aim)`) en turbo comme en
temps reel. Les 96 actions clavier et la baseline a 14 actions ne bougent
pas ; le score, le budget et la baseline sont ceux des phases precedentes.

**Mesurer sur des donjons apparies.** `Math.random` du jeu est remplace par
un generateur seede a l'ouverture de la page : deux runs de meme seed avec
la meme politique sont identiques pas a pas (verifie sur 1 500 pas, machine
chargee). Chaque essai se joue sur **80 donjons** (seeds 1000 a 1079) et
`scripts/compare_paired.py` mesure la difference donjon par donjon ; la
decision se prend sur les **PV perdus par salle** (somme sur somme, coup
fatal compris - l'audit l'oubliait) avec un intervalle a 95 % par
bootstrap. Un essai est adopte si l'intervalle exclut zero et si les salles
par episode ne reculent pas ; un essai neutre est retire.

### Le professeur, essai par essai

| Essai | Score | Salles / ep. | PV / salle [IC 95 %] | Victoires | Verdict |
| --- | --- | --- | --- | --- | --- |
| V45 professeur phase 4 | 358.81 | 12.2 | 1.032 [0.963, 1.108] | 2/80 | point zero |
| V46 visee a la souris | 552.42 | 18.4 | 0.836 [0.758, 0.917] | 7/80 | adopte |
| V47 zone morte, V48 anticipation, V49 priorite de cible | 515-559 | 17-19 | 0.83-0.86 | 5-6/80 | neutres |
| **V50 deplacement predictif** | **992.23** | **32.2** | **0.211 [0.184, 0.239]** | **64/80** | **adopte** |
| V51 corrections de blocage | 985.67 | 32.0 | 0.209 [0.185, 0.235] | 60/80 | neutre, conserve |
| V52 table des gardiens, V53 economie de build | 963-1002 | 31-33 | 0.18-0.21 | 59-63/80 | neutres |
| **Professeur gele, 80 donjons jamais vus (seeds 2000-2079)** | **1007.30** | **32.5** | **0.175 [0.150, 0.203]** | **60/80** | validation |

Deux essais font tout le chemin. La souris (V46) fait passer les tirs
capables de toucher de 41 % a la totalite et double les etages franchis. Le
**deplacement predictif** (V50, `src/mover.py`) remplace les regles
d'esquive par une simulation : les dix-sept deplacements que le clavier
permet (immobile, huit marches, huit dashs) sont joues sur 0,6 s a 20 Hz
contre ce que le jeu va faire - orbes en vol, zones qui eclatent a la fin
de leur meche, chargeur qui va foncer sur la droite qu'il a verrouillee,
bombe amorcee, tireur en preparation, attaque de gardien annoncee. Un
candidat touche est elimine ; parmi les autres, le moins cher pour le
combat gagne : bande de distance par famille (un chargeur se tient a 240-350
px parce que sa charge fait 170-218 px), ligne de vue, direction demandee,
et un dash qui coute cinq points quand une marche suffit. La securite est
une contrainte, jamais un objectif : maximiser le degagement envoyait le
heros dans les coins (V41-V42). Difference appariee V50 contre V49 : PV
par salle -0.653, intervalle [-0.731, -0.576], victoires 5 -> 64, meilleur
sur 75 donjons sur 80.

Les essais neutres sont documentes au carnet, dont une regression
instructive : les corrections de blocage (V51) ont d'abord fait passer les
victoires de 64 a 45 parce qu'un bug comptait « rester immobile » comme un
mur rencontre, et le heros ne pouvait plus attendre une piece aimantee.

### Ce qui est appris, et ce qui ne l'est pas

L'agent de la phase 5 est **hierarchique**. Un reseau appris decide le
pilotage - vingt actions : rester, huit directions, huit dashs, la
competence, la touche E. Deux reflexes explicites, **les memes fonctions
pour le professeur et pour l'eleve**, font ce qu'un joueur de twin-stick
fait de l'autre main et entre les combats : la visee (`src/aiming.py`, le
choix de la cible et l'angle du tir) et l'inventaire (`policies.
economy_action` : piedestal, achat, autel, portail, une trentaine de
decisions par run). Le clone est donc mesure sur le pilote qu'il a appris,
et l'ablation ci-dessous dit ce que cette partie apprise vaut.

Le vecteur d'observation passe a **439 features** (`src/features2.py`) :
tout ce que le professeur consulte y est - famille et affixe de chaque
ennemi, avancement de son telegraphe, recharge de sa prochaine attaque,
attaque du gardien par son nom, reliques portees, direction visee. Le
reseau (`src/dqn.py`) a **deux tetes**, direction et modificateur (marche,
dash, competence, E), des poids par classe pour les actions rares et un
lissage d'etiquettes contre un professeur parfois ambigu.

Ce n'etait pas assez. Le professeur choisit son deplacement en simulant les
0,6 s suivantes pour chacun des dix-sept mouvements du clavier ; un reseau
qui ne recoit que des positions et des vitesses doit refaire ce calcul de
tete, et ce sont ces pas-la qui coutent des coeurs. `src/features3.py`
ajoute le verdict de cette simulation - pour chaque deplacement : touche ou
non, dans combien de temps, quel degagement il garde - soit **507
features**. C'est le changement qui a fait la difference (V56), comme le
champ de danger a la phase 3. Le module est choisi par le champ
`feature_version` du modele, donc les clones plus anciens restent
rechargeables.

### Resultats

Protocole : 80 episodes x 9 000 steps, seeds 1000 a 1079, etage 1,
inventaire vide, sauvegarde vierge.

| Agent | Score | Salles / ep. | PV / salle | Etage 5 atteint | Victoires |
| --- | --- | --- | --- | --- | --- |
| Aleatoire (40 ep.) | 17.13 | 0.05 | - | 0/40 | 0/40 |
| Clone phase 4 (`bc_agent_v6`) | 249.49 | 8.9 | 1.049 | 0/80 | 0/80 |
| Clone phase 5 (`bc2_agent.json`, V54) | 636.42 +/- 30.9 | 20.9 | 0.893 | 20/80 | 7/80 |
| Clone phase 5 sans reflexe d'inventaire (ablation) | 430.10 +/- 19.4 | 14.7 | 0.650 | 0/80 | 0/80 |
| Clone phase 5 + DAgger (`bc2_agent_d1.json`, V55) | 660.09 +/- 28.8 | 22.0 | 0.821 | 20/80 | 10/80 |
| Clone features v3 (`bc3_agent.json`, V56) | 740.17 +/- 38.0 | 23.0 | 0.254 | 39/80 | 35/80 |
| **Clone features v3 + DAgger (`bc3_agent_d1.json`, V57)** | **921.36 +/- 24.6** | **29.5** | **0.292** | **58/80** | **49/80** |
| Professeur phase 5 (V51) | 985.67 +/- 17.7 | 32.0 | 0.209 | 70/80 | 60/80 |

L'agent retenu est le clone v3 apres une ronde DAgger : **921.36 sur 80
donjons, 29.5 salles par episode, 3.52 etages et 49 victoires completes sur
80** - 54 fois la baseline aleatoire du meme protocole, et 3,7 fois le clone
de la phase 4. Il est a 93 % de son professeur en score, 92 % en salles et
82 % de ses victoires.

Les deux dernieres lignes se lisent ensemble. Donner au reseau le verdict de
la simulation (V56) divise par trois ses PV perdus par salle, mais le rend
prudent au point de s'immobiliser : 36 de ses 80 runs epuisent leur budget
sans mourir. La ronde DAgger (V57) le remet en marche - +6.5 salles, +0.88
etage - sans lui rendre son imprudence. C'est aussi pourquoi le score seul
ne decide rien ici : V56 marque 740 points en restant souvent plante, le
temps de survie etant paye par `time * 0.1`.

L'ablation mesure ce que vaut le reflexe d'inventaire partage avec le
professeur : sans lui, le meme reseau ne depasse pas l'etage 4 et ne gagne
aucune run.

### Reproduction

```bash
# Point zero et mesures appariees (80 donjons, seeds 1000-1079)
python scripts/audit_deaths.py --policy heuristic --episodes 80 --workers 4 --seed 1000 \
    --output reports/death_audit_v45_zero_teacher.csv
python scripts/audit_deaths.py --policy teacher:lead,priority,mover --episodes 80 --workers 4 \
    --seed 1000 --output reports/death_audit_v51b_unstall.csv
python scripts/compare_paired.py reports/death_audit_v45_zero_teacher_episodes.csv \
    reports/death_audit_v51b_unstall_episodes.csv

# Le professeur, en direct
python scripts/watch_teacher.py --episodes 1 --seed 1021

# Demonstrations (300 episodes, curriculum d'etages), clonage, evaluation
python scripts/collect_demos2.py --teacher lead,priority,mover --episodes 300 --workers 6 \
    --floors 1 1 2 3 4 5 4 5 --seed 42 --output data/demos2_v1.npz
python scripts/train_bc2.py --demos data/demos2_v1.npz --epochs 30 --output models/bc2_agent.json
python scripts/audit_deaths.py --policy hybrid:models/bc2_agent.json --episodes 80 --workers 4 \
    --seed 1000 --output reports/death_audit_clone2.csv

# Features v3 : le verdict de la simulation, sur les memes demonstrations
python scripts/collect_demos2.py --teacher lead,priority,mover --episodes 300 --workers 6 \
    --floors 1 1 2 3 4 5 4 5 --seed 42 --features v3 --output data/demos3_v1.npz
python scripts/train_bc2.py --demos data/demos3_v1.npz --epochs 30 --output models/bc3_agent.json

# Ronde DAgger : le clone joue, le professeur etiquette, re-entrainement a chaud
python scripts/collect_demos2.py --policy models/bc3_agent.json --teacher lead,priority,mover \
    --episodes 100 --workers 6 --floors 1 1 2 3 4 5 4 5 --seed 77 --features v3 \
    --output data/dagger3_v1.npz
python scripts/train_bc2.py --demos data/demos3_v1.npz data/dagger3_v1.npz --epochs 8 --lr 3e-4 \
    --init models/bc3_agent.json --output models/bc3_agent_d1.json

# Evaluation au protocole, puis comparaison appariee avec le professeur
python scripts/audit_deaths.py --policy hybrid:models/bc3_agent_d1.json --episodes 80 --workers 4 \
    --seed 1000 --output reports/death_audit_clone3_v57_dagger.csv
python scripts/compare_paired.py reports/death_audit_clone3_v57_dagger_episodes.csv \
    reports/death_audit_v51b_unstall_episodes.csv
```

## Demo de l'agent entraine

L'agent se recharge depuis un script neuf, sans relancer l'entrainement, et
joue dans un navigateur visible :

```bash
# Agent retenu (phase 5, clone v3 + DAgger) - recommande pour la demo :
# la simulation des mesures, dessinee image par image au rythme du temps reel
python scripts/watch_agent2.py --agent models/bc3_agent_d1.json --episodes 3 --turbo --smooth

# Le meme en turbo, pour mesurer plutot que regarder
python scripts/watch_agent2.py --agent models/bc3_agent_d1.json --episodes 3 --turbo --headless

# Le meme en temps reel, par la boucle d'animation du jeu lui-meme
python scripts/watch_agent2.py --agent models/bc3_agent_d1.json --episodes 3

# Agent phase 4 (clone precedent)
python scripts/watch_dqn.py --episodes 3 --max-steps 9000

# Agent phase 1 (Q-learning tabulaire)
python scripts/watch_agent.py --episodes 3
```

Trois facons de regarder. En temps reel (sans `--turbo`), le jeu tourne sa
propre boucle d'animation et les touches passent par le navigateur avec une
latence variable : l'affichage est fluide mais la politique joue moins bien
que dans les mesures. En `--turbo`, chaque pas avance 18 frames de
simulation d'un coup puis dessine une seule image : c'est la simulation
exacte des mesures, mais vue a 7 images par seconde, le heros saute de
43 px entre deux images. `--turbo --smooth` garde ces 18 frames et les
dessine une par une, au rythme que l'ecran permet, avec un pas tenu a
0,15 s : memes decisions, meme partie, affichage fluide. Verifie sur la
graine 1021 : 600 pas joues en headless, en fenetre turbo et en fenetre
fluide donnent la meme trajectoire au pas pres (position, PV, kills,
salles), a 0,167 s par pas et 10,8 images par pas en fluide sur un ecran
60 Hz.

Pour que cette egalite tienne, le rendu ne tire plus dans les des du
combat : le dessin consomme `Math.random` (etincelle de la meche du
bombardier, tuile de grain au premier affichage), ce qui decalait les
tirages d'une partie regardee par rapport a la meme partie headless. Le
rendu a desormais son propre generateur, isole dans `__eosRender`.
`watch_teacher.py` accepte les memes options.

`watch_agent2.py` ne connait pas l'entrainement : il lit le JSON, choisit le
module de features d'apres le champ `feature_version` qu'il y trouve, et joue.
Il affiche la configuration rechargee avant le premier episode
(`features=507 (v3) actions=20 heads=True`).

Les poids ne sont pas versionnes (la consigne interdit de commiter des
donnees ou des modeles) : `models/bc3_agent_d1.json` se regenere avec les
commandes de reproduction de la phase 5, en environ trois heures dont
l'essentiel en collecte.

Note : les modeles des phases 2 a 4 ne sont plus rechargeables. L'observation
est passee de 109 a 202, 212, 439 puis 507 dimensions, et l'espace d'actions
de 59 a 64 puis 96 ; leurs resultats restent au carnet d'essais, mais la
demonstration rejouable est celle de la phase 5.

## Video de restitution

Lien video public : a ajouter apres upload.

Plan conseille pour la video de 10 a 15 minutes :

1. Presenter le jeu choisi et pourquoi il respecte les contraintes du projet.
2. Montrer la baseline aleatoire sur le protocole de la run complete et
   expliquer pourquoi le budget a change : une partie entiere coute environ
   9 000 steps. Dire aussi que la baseline elle-meme a du etre re-mesuree
   (34.11 sur 20 episodes, 17.13 sur 40) et pourquoi.
3. Expliquer les observations, les actions et la fonction de recompense, puis
   le contrat d'honnetete (sauvegarde vierge, evaluation depuis l'etage 1).
4. Expliquer le pipeline : professeur heuristique -> clonage -> DAgger, et
   pourquoi l'audit des morts pilote chaque decision.
5. Le coeur de la demonstration : ce qui a debloque chaque phase, en montrant
   a chaque fois la mesure qui l'a designe. Phase 4, la visee : le tir n'a lieu
   que 68 % du temps parce que l'espace d'actions etait incomplet, et seuls
   26 % des tirs peuvent geometriquement toucher parce que la visee clavier
   n'a que huit rayons (calcul `r x sin(22.5 deg)`). Phase 5, l'ecart entre le
   clone et son professeur : il tenait a une information absente du vecteur
   d'observation - le verdict de la simulation de deplacement - et non a la
   taille du reseau ni au nombre de demonstrations (V56).
6. Montrer la courbe `reports/progression_curve.svg` et le tableau comparatif,
   de la baseline (17.13) au clone retenu (921.36, 49 victoires sur 80).
7. Montrer une demo avec
   `python scripts/watch_agent2.py --agent models/bc3_agent_d1.json --episodes 3`,
   et dire ce qui est appris (le pilotage) et ce qui ne l'est pas (visee et
   inventaire, reflexes partages avec le professeur, chiffres de l'ablation
   V55 a l'appui).
8. Presenter quatre echecs instructifs du carnet : V29 (un defaut evident qui
   n'en etait pas un, -33 % de score), V34 (la passe RL qui ameliore le score
   en nettoyant 30 % de salles en moins), V42-V44 (la precision d'imitation
   monte de 81.7 a 84.1 % sans que le jeu bouge) et V56 (le clone cesse de
   prendre des coups... en cessant d'avancer : 36 runs sur 80 figees).
9. Conclure sur les limites et ce qui serait fait avec plus de temps.

## Essais et difficultes (pour le carnet d'essais)

Le carnet complet est disponible dans `docs/carnet_essais.md`. L'agent final est
le resultat de plusieurs iterations, chacune declenchee par un echec observe et
mesure :

1. **V1 aveugle aux portes** : l'etat n'encodait pas la direction de la sortie.
   En 50 episodes, zero salle terminee, plafond d'eval a 2.29 : dans une salle
   vide tous les points donnaient le meme etat, sortir etait inapprenable.
2. **V2 ping-pong** : apres ajout de la direction de porte, l'agent visait la
   porte la plus proche... qui est celle dans son dos quand il vient d'entrer.
   Allers-retours infinis entre salles nettoyees. Correctif : cibler par BFS la
   porte qui mene a la salle non visitee la plus proche.
3. **V3 colle aux murs** : la porte etait encodee par son mur (haut/bas/...),
   pas par sa direction relative - l'agent pressait "haut" sans s'aligner avec
   l'ouverture et se coincait contre le mur. Correctif : direction relative en
   8 secteurs (s'aligner d'abord, traverser ensuite).
4. **Politique deterministe qui se fige** : en jeu greedy pur, un obstacle
   invisible dans l'etat piegeait l'agent dans une boucle meme action -> meme
   blocage (l'exploration le decoincait pendant l'entrainement, plus rien en
   demo). Correctif : detecteur de blocage avec un coup aleatoire de sortie.
5. **Oscillation avec alpha constant** : dans les premiers runs, les evals
   s'effondraient apres chaque pic (11.39 -> 2.2) car la table courait toujours
   apres ses dernieres experiences. Correctifs combines : alpha decroissant
   0.25 -> 0.05 + meta-action de delegation (plancher heuristique). Resultat :
   plancher d'eval a ~6 au lieu de ~2, et un record qui monte de 10.64 a 15.20
   au fil des 800 episodes au lieu de s'eroder.
6. **Controle bas niveau enrichi** : les tirs en mouvement, tirs diagonaux et
   dashs directionnels ameliorent le comportement visuel, mais agrandissent trop
   l'espace si la Q-table doit tout apprendre directement.
7. **Intentions tactiques** : l'espace d'apprentissage a ete compacte en
   intentions (`fight`, `kite`, `exit`, `loot`, etc.) puis resolu en actions
   clavier. Les tests restent instables apres migration de l'ancien modele ; la
   version retenue est donc le Q-learning 16 actions avec controle bas niveau
   enrichi et garde-fous heuristiques.

Autres lecons : la penalite HP (-2/HP) peut rendre l'agent "froussard" apres
une serie de morts (observe puis resorbe avec plus d'episodes) ; les evals a
3 episodes restent bruitees car les donjons ne sont pas seedables (mitige par
le record = moyenne, et 32 points d'eval).

Avec plus de temps : davantage d'episodes d'eval par checkpoint, un deuxieme
run complet avec une autre seed pour comparer les courbes, et un petit DQN sur
les memes features pour generaliser entre etats voisins (le plafond actuel est
la granularite de la table, pas le volume d'entrainement).
