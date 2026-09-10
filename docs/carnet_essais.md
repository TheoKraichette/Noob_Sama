# Carnet d'essais

Ce carnet retrace les iterations mesurees pendant le projet. Le score compare
toujours la meme metrique :

```text
score = floors * 100 + rooms * 10 + kills + time * 0.1
```

La baseline de reference reste l'agent aleatoire sur 20 episodes de 100 steps :
score moyen 1.836, 0 kill et 0 salle terminee.

## Synthese

| Version | Changement teste | Signal observe | Decision |
| --- | --- | --- | --- |
| Baseline | Agent aleatoire | Score moyen 1.836, aucune progression de salle | Reference de comparaison |
| V1 | Q-learning sans direction de sortie | Environ 50 episodes, 0 salle terminee, plafond d'eval autour de 2.29 | Ajouter la direction de porte dans l'etat |
| V2 | Porte la plus proche | L'agent fait des allers-retours vers la porte par laquelle il vient d'entrer | Cibler la porte menant a la salle non visitee la plus proche |
| V3 | Direction de porte par mur | L'agent se colle aux murs sans s'aligner avec l'ouverture | Encoder la direction relative en 8 secteurs |
| V4 | Greedy deterministe | Blocage contre des obstacles non observes | Ajouter un detecteur de blocage en demo/eval |
| V5 | Alpha constant | Evals instables apres les pics | Decroissance alpha 0.25 -> 0.05 |
| V6 | Delegation heuristique | Plancher de perf plus stable, record eval 15.757 a ep. 975 | Conserver `heuristic` comme meta-action |
| V7 | Actions bas niveau enrichies | Plus de controle, mais espace d'action trop large si tout est appris directement | Garder les actions riches pour le controle heuristique |
| V8 | Intentions tactiques | Tests plus lisibles, mais les evals intent restent instables | Garder la version pratique Q-learning + garde-fous |

## Details des essais

### Baseline aleatoire

Commande :

```bash
python scripts/baseline_random.py --episodes 20 --max-steps 100 --seed 42
```

Resultat : score moyen 1.836, minimum 1.742, maximum 2.252. L'agent aleatoire
survit quelques secondes mais ne tue aucun ennemi et ne termine aucune salle.

### V1 - Etat aveugle aux portes

Premier probleme : apres un combat, toutes les positions d'une salle vide se
ressemblaient pour la Q-table. L'agent pouvait recevoir le bonus de salle, mais
il n'avait aucun signal d'etat pour apprendre ou se trouve la sortie.

Resultat observe : zero salle terminee sur les premiers essais courts, avec un
plafond d'evaluation autour de 2.29.

Correction : ajouter la direction de la porte cible dans l'etat discretise.

### V2 - Ping-pong entre salles

Apres ajout des portes, l'agent a commence a sortir des salles, mais il prenait
souvent la porte la plus proche. En entrant dans une nouvelle salle, cette porte
est souvent celle qui revient en arriere.

Resultat observe : allers-retours entre salles deja nettoyees.

Correction : calculer une porte cible par BFS dans le graphe du donjon pour
favoriser la salle non visitee la plus proche.

### V3 - Agent colle aux murs

Encoder seulement le mur de la porte (`up`, `down`, `left`, `right`) ne suffisait
pas. Si l'agent n'etait pas aligne avec l'ouverture, il pressait une direction et
restait bloque contre le mur.

Correction : encoder la direction relative vers la porte en 8 secteurs. L'agent
peut alors apprendre a s'aligner avant de traverser.

### V4 - Greedy deterministe bloque

En evaluation greedy, certaines positions non encodees dans l'etat provoquaient
des boucles : meme etat discret, meme action, meme blocage.

Correction : detecter 8 steps sans mouvement significatif hors combat et jouer
une action de sortie. Dans la version finale, cette sortie est deleguee a
l'heuristique plutot qu'a un coup aleatoire.

### V5 - Instabilite avec alpha constant

Les premiers entrainements montaient puis retombaient fortement, par exemple un
pic autour de 11 suivi d'une evaluation proche de 2. Le taux d'apprentissage
constant donnait trop de poids aux dernieres experiences.

Correction : faire decroitre alpha de 0.25 vers 0.05 pour stabiliser les acquis
en fin de run.

### V6 - Meta-action `heuristic`

L'exploration guidee seule aidait, mais l'agent ne savait pas toujours quand
laisser agir la politique de reference.

Correction : ajouter une action `heuristic` dans la Q-table. Le Q-learning reste
off-policy : il apprend a partir des coups joues, y compris ceux proposes par la
politique heuristique.

Resultat principal : meilleur agent a eval@975 avec score moyen 15.757, soit
environ 8.6 fois la baseline aleatoire.

### V7 - Controle bas niveau enrichi

Les observations visuelles ont montre que le controle simple etait limite en
combat. Des tirs diagonaux, tirs en mouvement et dashs directionnels ont ete
ajoutes.

Resultat : le controle heuristique devient meilleur, mais apprendre directement
toutes ces actions dans la Q-table agrandit trop l'espace d'action pour le budget
de deux jours.

Decision : garder ces actions dans le wrapper et dans les heuristiques, pas comme
espace principal d'apprentissage.

### V8 - Intentions tactiques

Derniere iteration experimentee : l'espace d'apprentissage est passe a des intentions
compactes (`fight`, `kite`, `dash_away`, `exit`, `loot`, `interact`, `wait`,
`heuristic`) resolues ensuite en actions clavier.

Resultat observe : les episodes d'entrainement peuvent etre bons, mais les
evaluations greedy restent instables apres migration de l'ancien modele. Cette
piste n'est donc pas retenue comme modele final.

Le record ponctuel `eval@1305` a 18.663 vient de la version pratique precedente :
modele Q-learning 16 actions, garde-fous heuristiques en evaluation/demo, et
controle bas niveau enrichi. L'evaluation finale annoncee dans le README reste
plus basse mais plus representative : moyenne 8.973 sur 20 episodes, avec des
runs faibles surtout dus aux combats ou l'agent perd trop de vie.

Decision : presenter la version finale comme un agent Q-learning tabulaire 16
actions avec controle bas niveau heuristique et garde-fous de demo, et garder
les intentions tactiques comme piste non retenue faute de stabilite.

## Phase 2 - environnement turbo, DQN et clonage de comportement

Objectif de la phase : depasser le plafond de la Q-table (granularite de la
table) et viser un agent capable de finir un etage. Budget re-echantillonne :
episodes de 900 steps, baseline aleatoire re-mesuree sur ce budget.

| Version | Changement teste | Signal observe | Decision |
| --- | --- | --- | --- |
| V9 | Environnement turbo (boucle du jeu pilotee en synchrone) | ~540 steps/s au lieu de 4-5, temps de jeu exact par step | Adopte ; baselines re-mesurees (600 steps : 8.45, 900 steps : 10.54) |
| V10 | DQN 59 actions sur 70 features continues | Le greedy converge vers "camper dans la salle de depart" (evals figees a 13.54) | Retirer le revenu de survie de la recompense d'entrainement |
| V11 | Temps retire + cout par step + epsilon d'eval | L'agent traverse les portes puis se fige/meurt en combat ; evals instables | Ajouter une meta-action de delegation comme en V6 |
| V12 | Delegation heuristique + penalite HP 1.0 | Record eval 25.4 (2.4x la baseline) mais le greedy n'utilise jamais la delegation (0/900 mesure) | Delegation apprise par sequences, pas par steps isoles |
| V13 | Delegation collante (sequences de 12 steps) + reprise du meilleur | 7 etages termines a l'ENTRAINEMENT (scores 147-181), mais ecart train ~64 / eval ~6 : la perf appartient aux sequences heuristiques | Mesurer honnetement chaque composante |
| V14 | Eval comparative 20 episodes x 900 steps | DQN pur 6.66 (sous l'aleatoire 10.54) ; heuristique corrigee 63.66 avec 2 etages | Resultat negatif documente : le RL pur n'apprend pas le combat fin sur ce budget |
| V15 | Clonage de comportement (94 074 paires heuristiques, 87.8 % de precision val.) | 56.89 de moyenne sur 20 episodes, 5.4x la baseline, 66 salles et 277 kills, reseau seul sans garde-fou | Le clonage marche ; ameliorer le professeur puis re-cloner |
| V16 | Esquives dans l'heuristique (projectiles, zones, patterns de boss exposes par le wrapper) | Audit des morts sur 20 episodes : 16 -> 9 morts, pertes face au gardien Vorace divisees par deux, eruptions 13 -> 6, ronces 22 -> 4 ; heuristique 63.66 -> 73.04 | Chaque parade validee par l'audit avant d'etre gardee |
| V17 | Re-clonage 100 features (117 151 paires) + une ronde DAgger (69 638 paires etiquetees sur les etats du clone) | 73.31 de moyenne, 7x la baseline, 2 etages en eval, ecart d'imitation referme (clone = professeur) | Le clonage transfere ; continuer d'ameliorer le professeur |
| V18 | Observation en demo : gel a l'entree des salles, pas de dash de voyage, routage aveugle a la carte. Correctifs : pre-positionnement pendant la fenetre d'apparition (le professeur figeait, le clone l'imitait), dash hors combat vers les cibles > 260 px, routage BFS par priorite (tresor > marchand > salles > mini-boss > gardien en dernier) | Professeur 73.04 -> 90.08 (6 morts/20 au lieu de 14) ; clone re-entraine (129 457 + 74 080 paires, 109 features) : 78.11 de moyenne, 7.4x la baseline | Une deuxieme ronde DAgger pour refermer l'ecart restant |
| V19 | Deuxieme ronde DAgger (74 484 paires sur les etats du clone V18, ~278 000 paires cumulees) | **93.23 de moyenne, 8.8x la baseline, 100 salles, 2 etages, pire episode a 51.2** - le clone egale son professeur (90.08) | Modele retenu : `models/bc_agent.json` (val acc 86.5 %) |
| V20 | Professeur plus agressif (distances de combat resserrees, fermeture en duel) | 80.9 de moyenne, 96 salles, mais 6 morts/20 au lieu de 2 : plus de kills, zero salle de gain | Essai negatif : reverti, les esquives paient plus que la proximite |
| V21 | Affinage RL depuis le champion (`scripts/train_dqfd.py` : perte TD sur l'experience fraiche + ancre supervisee sur ~370 000 demonstrations, lr 5e-5) | La politique derive (38) puis se reconstruit (~85 en mini-evals), mais deux checkpoints "records" testes au protocole complet plafonnent a 70.8 et 65.8, avec une regularite degradee (pire episode 2.96) | Essai negatif documente : le RL n'a pas depasse l'imitation dans ce budget ; le champion V19 reste le modele final |

### V10-V13 - trois pathologies du DQN, mesurees et corrigees

1. **Camping** : le terme de temps du score (+0.1/s) paye l'immobilite sans
   risque ; le greedy convergeait vers 13.54 exactement (survie totale, zero
   contact) a chaque eval. Correctif : temps retire de la recompense
   d'ENTRAINEMENT (le score de comparaison ne change pas).
2. **Gel devant la porte** : trace pas-a-pas : l'agent marche droit vers la
   porte puis se fige a 164 px, mal aligne, sur une action statique (etat
   inchange -> meme argmax -> boucle infinie). Correctifs : cout de -0.02 par
   step et 2 % d'exploration en eval (protocole DQN standard).
3. **Delegation morte** : la meta-action "heuristic" tiree step par step
   pendant l'exploration ne montre jamais la valeur de l'heuristique (un step
   isole dans du bruit). Correctif : sequences collantes de 12 steps. Resultat :
   des etages complets a l'entrainement, mais le reseau seul ne les a pas
   internalises dans le budget disponible.

### V15 - apprentissage par demonstrations

150 episodes heuristiques enregistres en turbo (94 074 paires
features -> action, 11 etages dans les demonstrations), reseau entraine par
entropie croisee (30 epoques, ~2 min), precision de validation 87.8 %.

Le reseau clone atteint ~89 % du score de son professeur et nettoie 2 a 5
salles a chaque episode du protocole. Sa limite est celle du professeur : le
combat de boss. D'ou V16 : ameliorer le professeur, puis re-cloner (ses
progres se transferent par re-collecte).

### V16 - le professeur apprend a esquiver

L'audit des causes de mort (lecture de `killerName` dans le jeu a chaque
perte de vie) a montre que 16 morts sur 20 arrivaient dans la salle du boss,
avec le gardien "Vorace" pour premier tueur (46 pertes de vie), devant les
projectiles (22) et les eruptions telluriques (17). Trois reponses, chacune
re-mesuree par le meme audit :

1. Wrapper : exposition des projectiles ennemis (`enemyShots`), des zones
   d'explosion (`hazards`, avec meche restante) et de l'attaque en cours des
   boss (`attackName`, `atkStep`).
2. Heuristique : pas de cote perpendiculaire aux orbes (point d'approche
   minimale), sortie des cercles annonces (dash si la meche est courte),
   strafe des assauts re-vises (`tripleLunge`), distance pendant le
   `slamQuake`, punition du boss etourdi apres un `heavyCharge` manque.
3. Choix de direction par scoring des 8 directions contre TOUS les dangers
   connus (zones, ennemis proches, murs) : fuir une eruption ne doit pas
   mener dans une autre, ni dans une ronce.

Resultat : 16 -> 9 morts sur 20 episodes, pertes face au Vorace divisees par
deux, eruptions 13 -> 6, ronces 22 -> 4 ; score moyen 63.66 -> 73.04 et un
plancher qui monte de 23.8 a 40.1.

### V17 - re-clonage et DAgger

Re-collecte avec le professeur ameliore : 117 151 paires (100 features
desormais, menaces incluses — sans elles le clone ne pourrait pas expliquer
les esquives qu'il imite), 25 etages dans les 150 episodes. Clonage 256x256
sur 60 epoques : 84.6 % de precision, 66.57 en eval.

Puis une ronde DAgger : le clone joue 100 episodes, l'heuristique etiquette
chaque etat visite (69 638 paires). Le reseau re-entraine sur l'union
apprend a corriger ses propres derives — la faiblesse classique du clonage
pur. Precision 85.6 %, et surtout : **73.31 de moyenne sur le protocole
(7x la baseline), 2 etages termines en eval, ecart clone/professeur
referme**.

Piste restante pour depasser le professeur : affinage RL depuis ce modele
(type DQfD, perte supervisee + perte TD melangees — `train_step_bc` et
`train_step` coexistent deja dans `src/dqn.py`).

Commandes de reproduction :

```bash
python scripts/collect_demos.py --episodes 150 --max-steps 900 --seed 42
python scripts/train_bc.py --hidden 256 256 --epochs 60
python scripts/collect_demos.py --policy models/bc_agent.json --episodes 100 --seed 7 --output data/dagger1.npz
python scripts/train_bc.py --demos data/demos.npz data/dagger1.npz --hidden 256 256 --epochs 60
python scripts/watch_dqn.py --agent models/bc_agent.json --episodes 3
```

## Phase 3 - viser la run complete

Changement d'objectif : ne plus battre la baseline, mais **terminer le jeu**,
c'est-a-dire atteindre l'etat `victory`. La lecture du code du jeu donne la
taille reelle de la cible : `FINAL_FLOOR = 5`, etages de 8/9/10/11/8 salles,
soit 46 salles et 5 gardiens, ~15-19 min de jeu annoncees par les auteurs.
A 0.13 s de jeu par step, une run complete coute environ **9 000 steps** : le
budget de 900 de la phase 2 en couvrait un huitieme, la victoire y etait
impossible par construction, pas par faiblesse de l'agent.

Nouveau protocole : **20 episodes x 9 000 steps**, baseline aleatoire
re-mesuree dessus : **34.11** (min 2.13, max 134.95, ecart-type 32.83 ;
`reports/random_baseline_9000.csv`). L'aleatoire meurt dans 19 cas sur 20 -
il ne campe plus, il sort de la salle de depart et se fait tuer.

Trois corrections de protocole accompagnent le changement :

- **Sauvegarde vierge a chaque episode.** Le jeu conserve une meta-progression
  dans `localStorage` (heros debloques, eclats, Sanctuaire). Elle s'accumulait
  silencieusement entre les episodes d'une meme session : les derniers
  tournaient sur un personnage plus fort que les premiers, et que la baseline.
  Le stockage est desormais vide a chaque chargement de page (verifie :
  `runs = 1`, `shards = 0`, aucun heros debloque a chaque reset).
- **Curriculum d'entrainement.** `reset(start_floor=n)` demarre un episode
  d'ENTRAINEMENT a un etage donne avec un build plausible, tire par le
  generateur d'objets du jeu lui-meme. Interdit en evaluation, ou une
  assertion verifie etage 1 et inventaire vide.
- **Audit des morts** (`scripts/audit_deaths.py`) : une ligne par point de vie
  perdu, avec le coupable nomme par le jeu (`killerName`), l'etage, la salle
  et l'attaque de boss en cours. Toutes les decisions ci-dessous en viennent.

### Le professeur, mesure par mesure

| Version | Changement teste | Score | Signal | Decision |
| --- | --- | --- | --- | --- |
| Baseline | Agent aleatoire, 20 x 9 000 | 34.11 | 19/20 morts, 0 salle | Reference |
| V21 | Etat de la phase 2, re-mesure sur 9 000 | 148.01 | 19/20 morts, 13 runs meurent des l'etage 1 ; 111 des 149 PV perdus a l'etage 1 | Point de depart |
| V22 | Champ de danger : 16 directions x 3 distances, agrege sur TOUS les dangers cote JS | 174.34 | Morts 19 -> 14, PV perdus 149 -> 117, eruptions 15 -> 7, ronces 12 -> 8 | Adopte |
| V23 | Reponses par attaque de gardien (14 attaques ajoutees) | 189.24 | Etage 3 atteint pour la premiere fois, blocages 6 -> 1 - mais PV 117 -> 154 et `heavyCharge` 14 -> 28 | Adopte, timing a corriger |
| V24 | Timing : dasher APRES le verrouillage de visee, pas pendant | 208.46 | Vorace 40 -> 20 PV (`heavyCharge` 28 -> 16, `tripleLunge` 12 -> 4), 0.82 PV perdu par salle, etage 2 dans 12/20 | Adopte |
| V25 | Anti-blocage + ligne de vue sur les piliers + competence du heros + portee de ramassage | 225.59 | **Blocages 5/20 -> 0/20**, aucun budget epuise, 165 salles au lieu de 141 - mais 1.04 PV par salle | Adopte, competence a restreindre |
| V26 | Build et economie : ramasser le butin des salles scellees, autel du sacrifice, choix d'objet par rarete, boutique qui garde l'or pour les coeurs ; competence restreinte | **272.83** | Etages atteints 1:4 2:13 3:3 (contre 1:8 2:10 3:2), 185 salles, meilleur episode 683.9 | Adopte |
| V27 | Esquive dimensionnee : dash calcule sur la distance a couvrir, traversee d'un mur d'orbes | 252.30 | Aucun gain, 1.02 PV par salle | **Essai negatif, reverti** |
| V28 | Retrait des seules regles qui consomment du dash | **272.06** | **1 victoire sur 20**, meilleur episode 1038.7, 0.99 PV par salle (record), 0 blocage | **Adopte - professeur retenu** |
| V29 | Garder la moitie cardinale d'un deplacement diagonal au lieu de tirer sur place | 183.69 | 1.19 PV par salle, 12 runs sur 20 meurent des l'etage 1 | **Essai negatif, reverti** |

### V22 - ce que l'agent ne pouvait pas voir

L'observation tronquait les menaces a `shots[:6]` et `hazards[:6]`. Or
`aftershock` pose 19 zones d'un coup, `frostNova` 16, `tidalPulse` tire 27
orbes : pendant les attaques qui tuent, l'agent voyait un tiers du danger.

Relever les plafonds ne suffisait pas - une liste de 20 zones n'est pas une
representation exploitable par un reseau de taille fixe. L'environnement
calcule donc un **champ de danger** : pour 16 directions et 3 distances de
projection, un score agrege sur toutes les zones (en tenant compte de leur
meche restante), tous les projectiles (point d'approche minimale sur leur
trajectoire), tous les ennemis et les murs. 48 flottants qui rendent
representable ce qu'une liste tronquee ne pouvait pas decrire. Le professeur
consomme le meme champ, ce qui a supprime au passage une duplication de la
logique de scoring cote Python.

### V24 - une parade peut couter plus cher que l'absence de parade

V23 a fait doubler les degats de `heavyCharge` alors qu'une reponse y etait
codee. La lecture du registre d'attaques du jeu donne la raison : le gardien
**suit le joueur jusqu'a `atkT = 0.5`**, verrouille, puis charge a 720 px/s a
partir de 0.75. La parade s'declenchait a 0.35 - pendant la phase de suivi.
Le gardien re-visait la nouvelle position, et le dash etait en recharge
(0.85 s) au moment ou la charge partait vraiment.

Attendre le verrouillage a divise les degats du gardien par deux. La lecon
n'est pas « il fallait dasher » mais « il fallait dasher au bon moment », et
ce moment n'etait pas observable avant que `attack_time` ne soit expose.

### V25 - le blocage, un cout invisible dans la table des morts

L'audit ne comptait que les degats. Or six episodes sur vingt epuisaient leur
budget sans mourir : l'episode 17 de V22 a survecu 22 minutes de jeu en ne
nettoyant que 5 salles. En ajoutant un compteur de steps sans progression, le
diagnostic est tombe : jusqu'a **7 380 steps immobiles** - 82 % d'un budget -
toujours dans une salle de type `combat`.

Cause trouvee dans `room.js` : les salles de combat a partir de la profondeur 2
contiennent des **piliers**, qui bloquent les deplacements *et* les
projectiles (`Room.projectileBlocked`), et le biome Bibliotheque en aligne des
rangees entieres. Rien de tout cela n'etait expose. L'agent tirait dans un
pilier, indefiniment, avec l'ennemi derriere.

Correctifs : exposition des piliers, calcul d'une ligne de vue par ennemi,
prise en compte des piliers dans le champ de danger, deplacement lateral quand
la ligne de tir est bouchee, et detecteur de blocage generique. Blocages :
5/20 -> 0/20.

### V26 - le professeur payait le prix des salles sans en prendre la recompense

La politique refusait toute interaction dans les salles dites « risquees »
(`altar`, `defi`, `scelle`, `gambler`). En lisant `game.js`, cette regle
melangeait deux choses tres differentes :

- **declencher** une epreuve optionnelle (le brasier chronometre, le parieur,
  l'autel qui coute un coeur) - un vrai risque ;
- **ramasser** un butin deja pose sur un piedestal - aucun risque du tout.

La salle scellee fait combattre deux elites renforcees et depose une relique
epique au nettoyage. Le professeur menait le combat entier, puis repartait
sans la relique : tout le cout, aucune recompense. Meme chose pour la salle de
defi.

Avec le butin repris, le choix d'objet classe par rarete (prendre un objet sur
un piedestal fait disparaitre les autres : le plus proche est rarement le bon),
l'autel accepte quand la barre de vie le permet, et une boutique qui garde
l'or pour les coeurs quand la vie manque : **225.6 -> 272.8**, et le nombre de
runs qui meurent des l'etage 1 tombe de 8 a 4.

### V27-V28 - le dash est un budget, pas un reflexe

V27 ajoutait deux regles qui dashent davantage : sortir d'une zone quand le
calcul montre que la marche ne suffit pas, et traverser un mur d'orbes a
l'invulnerabilite. Les deux sont justes prises isolement, et pourtant le score
est descendu de 272.8 a 252.3.

Explication : le dash a **0.85 s de recharge**. Chaque dash depense sur un
orbe est un dash absent sous la charge du gardien, qui est l'attaque la plus
chere du jeu. En retirant ces deux regles et en gardant les parties gratuites
de V27 (choisir la zone a la meche la plus courte, offrir le recul comme
troisieme option d'esquive), le score revient a 272.1 avec le meilleur taux de
degats jamais mesure - **0.99 point de vie perdu par salle** - et la
**premiere victoire** du projet.

Episode victorieux : 5 etages, 32 salles, 227 ennemis vaincus, **13 objets**,
score 1038.7. Les runs qui meurent a l'etage 1 en ont 1. Le build n'est pas un
detail de confort, c'est la variable qui decide de la run.

### V29 - un defaut evident qui n'en etait pas un

L'espace d'actions ne contient pas de combinaison deplacement-diagonal + tir.
Face a une direction diagonale, `_combine` abandonnait donc le deplacement et
tirait sur place - s'immobiliser en plein combat, ce qui ressemble beaucoup a
un defaut. Garder la moitie cardinale la moins dangereuse semblait
strictement meilleur.

Mesure : **272.1 -> 183.7**, et douze runs sur vingt meurent des l'etage 1 au
lieu de six. La moitie conservee n'eloigne pas de la menace, elle longe : un
recul « bas-gauche » reduit a « bas » suit l'ennemi au lieu de le fuir. Tirer
sur place garde au moins les degats sur la cible.

Essai reverti. C'est le meilleur exemple du carnet qu'un raisonnement
convaincant ne remplace pas une mesure.

### Sonde de profondeur - ce que le curriculum revele

`audit_deaths.py --start-floor 4` fait jouer le professeur directement a
l'etage 4 avec un build plausible. Resultat : **12 episodes sur 12 se
terminent par une mort**, 16 salles nettoyees au total, et **3.38 points de
vie perdus par salle** contre environ 1.0 aux etages 1 et 2.

C'est une mesure, pas une impression : la difficulte de fin de run
(vie des ennemis x4.2, degats de contact +1, tirs a 2 points) est d'un autre
ordre. Le plafond du professeur n'est donc pas « arriver a l'etage 4 » mais
« y survivre ».

### L'agent appris

Meme protocole, meme score, meme budget. Les colonnes « salles » et « etages »
sont des totaux sur les 20 episodes.

| Version | Agent | Score | Salles | Etages | Blocages | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| V30 | Clonage sur 243 751 paires (180 episodes, dont un tiers en curriculum d'etage) | 186.95 | 117 | 9 | 5/20 | Score flatteur, voir ci-dessous |
| V31 | + une ronde DAgger (193 116 paires) | 159.02 | 127 | 8 | **0/20** | Adopte malgre le score |
| V32 | Auto-imitation iteree, 10 rondes de 48 episodes | 158.36 | 131 | 8 | 0/20 | Essai neutre |
| V33 | + une deuxieme ronde DAgger (144 571 paires) | **181.42** | **137** | **10** | 2/20 | **Modele retenu** |
| V34 | Affinage RL (DQfD) depuis le champion d'auto-imitation, 160 episodes | 198.04 | 97 | 5 | 13/20 | **Essai negatif : le score monte, le jeu recule** |

### V30-V31 - quand le score recompense l'immobilite

Le premier clone marque 186.95, le clone corrige par DAgger seulement 159.02.
En metriques de jeu, le second est pourtant nettement meilleur : **127 salles
au lieu de 117, en moins de la moitie des steps** (29 885 contre 73 493) et
sans un seul blocage.

L'ecart vient du score lui-meme. Le terme `time * 0.1` paie 0.1 point par
seconde survecue : un episode fige pendant tout son budget rapporte environ
135 points de pure survie - davantage qu'une run honnete qui nettoie cinq
salles et meurt. Cinq des vingt episodes du premier clone sont dans ce cas.

C'est la pathologie de « camping » deja rencontree en V10, revenue par une
autre porte. Le score n'a pas ete change : c'est le referentiel de comparaison
avec la baseline, et le modifier obligerait a tout re-mesurer. En revanche,
deux precautions ont ete prises :

- l'audit et les tableaux affichent desormais **salles et etages** a cote du
  score, parce que ces deux-la ne se gagnent pas en restant immobile ;
- l'auto-imitation classe ses episodes sur la progression avant le score.
  Sans cela, elle aurait appris a se figer, la meilleure strategie que le
  score recompense.

### V32 - l'auto-imitation, et l'importance de savoir refuser

Principe : l'agent joue, on ne garde que le quart superieur de ses episodes,
et on le re-entraine sur ce qu'il a fait dans ceux-la. C'est le mecanisme qui
peut depasser le professeur, puisqu'il n'imite plus personne.

Premiere version, echec net : la progression moyenne descend de 12.29 a 8.94
puis 8.21 en trois rondes. Deux causes, toutes deux corrigees :

1. La boucle avancait sans jamais **refuser** une ronde. Une mauvaise ronde
   devenait le point de depart de la suivante, et l'erreur se composait.
   Correctif : un pas qui ne joue pas mieux est annule, et la ronde suivante
   repart du champion.
2. Le pas d'apprentissage (2e-4, 6 epoques sur ~30 000 paires) sur-ajustait un
   petit jeu de donnees d'elite. Correctif : 5e-5, 3 epoques, et autant de
   paires de demonstration re-injectees que de paires fraiches.

Deuxieme version : progression 12.10 -> **14.67** a la ronde 2, puis huit
rondes consecutives refusees. Le champion de la ronde 2 marque 213.78 sur ses
48 episodes de rollout - mais **158.36 au protocole complet**, soit
exactement le niveau du modele de depart.

Conclusion honnete : le gain apparent etait du bruit d'echantillonnage sur
48 episodes, et selectionner sur une mesure bruitee revient a selectionner du
bruit. L'auto-imitation n'a pas depasse l'imitation dans ce budget. Elle
laisse tout de meme le meilleur rythme de progression mesure
(4.59 salles pour 1 000 steps contre 4.25).

### V33 - la deuxieme ronde DAgger

Comme en phase 2, c'est la deuxieme ronde qui compte. Le champion de
l'auto-imitation joue 120 episodes, l'heuristique etiquette chacun de ses
etats (144 571 paires), et le reseau se re-entraine sur les trois jeux reunis
(581 438 paires, precision de validation 83.7 %).

Resultat : **181.42**, 137 salles, 10 etages franchis, et le premier clone a
atteindre l'etage 3. C'est le modele retenu.

L'ecart au professeur (272.06) ne se referme pas comme en phase 2. Deux
raisons mesurables : l'espace d'actions est passe de 59 a 64 et l'etat de 109
a 202 dimensions, ce qui rend l'imitation plus difficile a precision egale ;
et le professeur tire desormais une part de sa force de decisions **rares**
(prendre l'autel, classer trois reliques, garder l'or pour un coeur) qui
n'apparaissent que quelques dizaines de fois dans 580 000 paires.

### V34 - la passe RL, ou le score contre le jeu

Affinage type DQfD depuis le champion d'auto-imitation : perte TD sur
l'experience fraiche, ancre supervisee sur les 437 000 demonstrations,
lr 5e-5, penalite de vie 1.5, un pas de gradient toutes les 4 transitions
(sans quoi l'entrainement coute plus cher que la simulation). 160 episodes de
9 000 steps, environ 4 h 30.

La courbe d'evaluation monte franchement : 181.0 -> 179.7 -> 214.5 -> **262.0**
a l'episode 160. Mesure au protocole complet de 20 episodes, le meme point de
controle donne **198.04** - au-dessus du clone retenu (181.42). Le RL semble
donc avoir gagne.

Il n'a pas gagne. Les metriques de jeu du meme fichier :

| | Score | Salles | Etages | Episodes a 9 000 steps | Salles / 1 000 steps |
| --- | --- | --- | --- | --- | --- |
| Clone V33 | 181.42 | **137** | **10** | 2/20 | **3.19** |
| RL V34 | 198.04 | 97 | 5 | **13/20** | 0.70 |

L'agent RL nettoie **30 % de salles en moins** et franchit deux fois moins
d'etages, tout en marquant plus. Il a appris a survivre sans jouer : sa
recompense d'entrainement penalise les degats (-1.5 par point de vie) et paie
peu la salle (+1 apres mise a l'echelle), donc l'optimum est d'eviter tout
contact ; et le score de comparaison, lui, paie 0.1 par seconde survecue. Les
deux se sont accordes contre le jeu.

C'est la troisieme apparition du camping dans ce carnet (V10, V30, V34), et la
plus instructive : cette fois il se cache derriere une amelioration du score.
Sans les colonnes « salles » et « etages », l'essai aurait ete adopte.

Modele retenu a l'issue de la phase 3 : **`models/bc_agent_v5.json`**, le
clone a deux rondes DAgger. La phase 4 lui succede.

## Phase 4 - la ou le professeur perdait vraiment ses coeurs

Deux constats de depart, tous deux tires de la lecture du jeu et de l'audit,
et non d'une impression :

1. **Le joueur a 6 demi-coeurs** (`PLAYER_CFG.maxHp`), et le professeur en
   perdait 1.0 par salle. Une run entiere fait 46 salles. Toute la partie se
   joue donc sur les degats encaisses, pas sur le score.
2. **171 des 261 coups recus l'etaient sans un seul projectile a l'ecran**, et
   92 avec un seul ennemi present. Ce n'etait pas un probleme de perception ni
   d'esquive : le professeur se faisait toucher au corps a corps par des
   ennemis deux fois plus lents que lui.

Le protocole de mesure change aussi : 20 episodes laissent une erreur type de
~25 points sur la moyenne, qui masque tout ce qui est plus petit qu'elle.
`audit_deaths.py` prend un `--workers`, les essais passent a 40 episodes, et
la metrique de decision devient les **PV perdus par salle** — elle compte des
centaines d'evenements au lieu de 40, donc son erreur est bien plus serree.

### V35 - l'espace d'actions etait incomplet

`js/core/input.js` resout le deplacement (`getMove`) et la visee (`getShoot`)
sur deux axes **independants**. Maintenir W + A + Fleche-haut est donc une
entree parfaitement legale — et il n'existait aucune action « diagonale +
tir » dans les 64 disponibles. Verifie dans le navigateur avant d'y toucher :
`vx=-205 vy=-205` (norme 290) avec `aim=[1,0]`.

Consequence, mesuree sur 5 341 pas de combat reels : le professeur ne tirait
que pendant **68,3 %** d'entre eux. Des que la direction de fuite etait
diagonale — une fois sur deux — il lachait le tir et s'immobilisait, face a
des poursuivants a 125 px/s quand lui en fait 290.

32 actions ajoutees (4 diagonales x 8 directions de tir), 96 en tout. Le temps
de tir monte a **92,6 %**.

Resultat : **250.94** sur 40 episodes, et une regression cachee dans le
detail : les degats de `heavyCharge` passent de 0,55 a 1,6 PV par episode.
J'avais profite du changement pour ouvrir les huit directions au strafe
defensif. Ce qu'une charge demande, c'est du terrain gagne **en travers** de
sa ligne, et une diagonale n'en couvre que 0,707. Strafe defensif remis aux
quatre cardinales.

### V36 - le tir pendant l'esquive

Meme trou, applique aux esquives : `_hazard_escape`, `_shot_dodge` et les
reponses aux attaques de gardien ne tiraient que si leur pas etait cardinal.
Or l'esquive occupe l'essentiel d'un combat de gardien.

Resultat : **228.09 +/- 16.24**, 1,11 PV perdu par salle. Rien de gagne. Le
tir supplementaire ne protege de rien par lui-meme — il fallait d'abord qu'il
touche.

### V37 - viser, c'est tourner autour de la cible

La mesure qui explique tout le reste. Sur 4 948 tirs reels, **seuls 25,6 %
pouvaient geometriquement atteindre leur cible** ; ecart lateral median
52 px pour un corps et un orbe qui font ensemble 20 px.

La cause est arithmetique : la visee clavier n'a que huit rayons, 45 degres
d'ecart. Une cible a un demi-secteur du rayon est manquee de
`r x sin(22.5 deg)`, soit 115 px a 300 px de distance. Il faut tomber a moins
de 3,8 degres pres pour toucher a cette distance — environ un tir sur six.

Et c'est exactement la que l'audit trouvait ses degats : entre 330 et 470 px,
avec un seul ennemi et de la vie, l'ancienne regle repondait « rester
immobile et tirer ». Rester immobile n'esquive rien et ne vise rien.

Correctif : quand la cible n'est pas sur un rayon, **tourner autour d'elle**
jusqu'a ce qu'elle y soit, sans jamais cesser de tirer. Le strafe defensif
choisit desormais, entre ses deux perpendiculaires, celle qui referme l'angle.

| | Score | Salles / ep. | PV / salle | Etage max | Victoires |
| --- | --- | --- | --- | --- | --- |
| V36 | 228.09 | 8.05 | 1.11 | 2 | 0/40 |
| **V37** | **307.33** | **10.55** | **0.92** | **5** | **1/40** |

Tirs pouvant toucher : 25,6 % -> **40,8 %**. Ecart lateral median 52 -> 31 px.
Un seul blocage sur 40 episodes contre quatre.

### V38 - anticiper le mouvement de la cible : essai negatif

Suite logique : un orbe met `distance / 580` a arriver, soit une demi-seconde
a 300 px, pendant laquelle la cible quitte sa propre boite de collision. Les
vitesses ennemies ont donc ete exposees (`vx`, `vy`) et la visee portee sur le
point d'impact plutot que sur la position courante.

La visee s'ameliore bien : 42,7 % de tirs pouvant toucher, et surtout une
queue de distribution qui s'effondre (q90 de 179 a 82 px). **Le jeu, lui, se
degrade** : 1,01 PV perdu par salle contre 0,92, plus aucune victoire, etage 3
au mieux.

Explication : les poursuivants accelerent a 850 px/s2 et oscillent
(`wobble`). Leur vitesse instantanee predit mal une demi-seconde, et comme le
point vise pilote aussi le **deplacement** (c'est lui qu'on encercle), une
prediction bruitee fait tourner la politique dans le vide. Anticipation
abandonnee ; `vx` et `vy` restent dans les features, ou le reseau peut en
faire ce qu'il veut.

### V39 - choisir le bon rayon

En separant les deux sous-changements de V38, il restait une correction pure.
`_shoot_towards` choisissait une diagonale des que les deux ecarts depassaient
45 px : une cible a 14 degres de l'horizontale, mais a plus de 45 px sur les
deux axes, se voyait tirer dessus a 45 degres. Le rayon le plus proche du
relevement est desormais choisi directement.

**347.06 +/- 36.86**, 12,1 salles par episode, **0,84 PV perdu par salle**,
aucun blocage sur 40 episodes, et les cinq etages atteints (1:8 2:19 3:8 4:3
5:2) avec une victoire.

### V40 - aller chercher le soin de plus loin : essai negatif

L'economie du soin est le vrai plafond : six demi-coeurs pour quarante-six
salles, une gemme sur trente ennemis, et neuf secondes avant qu'elle ne
s'evapore. En combat, le professeur ne se detournait que pour un coeur a
moins de 140 px. Porte a 340 px des que la barre est sous la moitie.

**321.08**, 0,89 PV perdu par salle, aucune victoire : moins bien que V39 sur
toutes les colonnes. Traverser une salle en combat pour un demi-coeur coute
plus de vie qu'il n'en rend. Revert, et la borne d'origine est confirmee par
la mesure plutot que par l'intuition qui l'avait posee.

### Bilan de la sequence

| Version | Score | Salles / ep. | PV / salle | Etage max | Victoires |
| --- | --- | --- | --- | --- | --- |
| Reference V34 (20 ep.) | 272.06 | 9.7 | 0.99 | 5 | 1/20 |
| V36 tir pendant l'esquive | 228.09 | 8.05 | 1.11 | 2 | 0/40 |
| V37 encerclement | 307.33 | 10.55 | 0.92 | 5 | 1/40 |
| V38 anticipation | 290.77 | 10.45 | 1.01 | 3 | 0/40 |
| **V39 bon rayon** | **347.06** | **12.1** | **0.84** | **5** | **1/40** |
| V40 soin de loin | 321.08 | 11.5 | 0.89 | 4 | 0/40 |

Le chiffre de 272.06 de la phase 3 vient de 20 episodes seulement, dont
**une victoire a 1038 points** qui pese a elle seule ~39 points sur la
moyenne : il n'est pas comparable terme a terme avec les mesures a 40
episodes ci-dessus. C'est precisement pour cela que le protocole d'essai est
passe a 40 episodes et que la decision se prend sur les PV perdus par salle.

La meme prudence s'applique a la baseline elle-meme. Re-mesuree sur 40
episodes, elle donne **17.13** (mediane 12.3) au lieu des 34.11 mesures sur
20 : deux episodes campes a 135 et 109 points portaient a eux seuls la moitie
de l'ancienne moyenne. Les rapports « x fois la baseline » de ce depot
continuent d'utiliser **34.11**, le chiffre le plus haut, donc le plus
defavorable a l'agent.

### V41 a V44 - le clone, et un plafond qui ne bouge pas

Tout est recollecte : 96 actions, 212 features (les stats de tir du joueur,
les vitesses ennemies et l'ecart de visee ont ete ajoutes). 350 episodes du
professeur, dont 150 avec curriculum en profondeur -> **553 763 paires**.

Les 32 actions ajoutees representent **21,8 %** des decisions du professeur :
elles ne sont pas decoratives. Le motif le plus parlant du jeu de donnees est
`down_left_shoot_up_right` — reculer en diagonale en tirant derriere soi,
c'est-a-dire l'esquive-tir qui etait litteralement inexprimable avant.

| Version | Donnees | Reseau | Precision val. | Score (40 ep.) | PV / salle | Blocages |
| --- | --- | --- | --- | --- | --- | --- |
| **V41 clone** | 554 k | 384x384 | 81.7 % | **278.07** / 251.57 | 0.82 / 0.92 | **0/80** |
| V42 + 1 DAgger | 726 k | 384x384 | 82.8 % | 254.83 | 0.87 | 2/40 |
| V43 + 2 DAgger | 917 k | 384x384 | 83.7 % | 264.94 | 0.95 | 2/40 |
| V44 + 2 DAgger | 917 k | 512x512 | 84.1 % | 260.59 | 0.90 | 1/40 |

**La precision d'imitation monte de 81.7 a 84.1 %, et le jeu ne bouge pas.**
Les quatre modeles tiennent dans un intervalle de 25 points alors que leurs
erreurs types valent 18 a 22 : ils sont indiscernables. Ni les rondes DAgger
ni le reseau plus large n'ont produit de gain mesurable, contrairement a la
phase 2 ou la deuxieme ronde DAgger avait compte.

Deux mesures rappellent au passage que 40 episodes ne suffisent toujours pas
a departager des politiques proches : le meme modele V41, mesure deux fois
sur 40 episodes, donne 278.07 puis 251.57. C'est la raison pour laquelle il
est retenu sur ses **80 episodes cumules** (264.8) et sur la seule difference
qui sorte du bruit : aucun blocage, quand tous les autres en ont.

Essai complementaire : evaluation en greedy pur (`--epsilon 0`) plutot qu'avec
2 % d'exploration. Le score ne bouge pas (263.75) mais les PV perdus par
salle tombent a **0.76**, le meilleur chiffre de tout le projet, professeur
compris — au prix de deux blocages en salle au tresor, exactement ce que les
2 % d'exploration existent pour eviter. Configuration d'evaluation conservee
a 0.02.

Modele final retenu : **`models/bc_agent_v6.json`**, le clone par imitation
seule, sur ses 80 episodes cumules (264.8, 0.87 PV par salle, aucun blocage).
Sur le protocole historique de 20 episodes il marque **242.42**, contre
181.42 pour le champion de la phase 3, avec 169 salles au lieu de 137 et
aucun episode fige.

## Phase 5 - viser juste, esquiver vraiment, finir la run

Le point de depart est l'audit de 40 episodes du professeur de la phase 4
(`reports/death_audit_v44_life.csv`), lu autrement : sur 420 PV perdus,
**47 % le sont au corps a corps pur** (aucun projectile, aucune zone a
l'ecran) et **164 PV le sont avec un seul ennemi present**. Le deuxieme
tueur du jeu est l'Ame errante, l'ennemi le plus lent du bestiaire (125 px/s
contre 290 pour le heros). Le dernier coup d'une run tombe au step median
~1 900 sur un budget de 9 000 : les runs meurent tot, le budget est inutilise.

La cause n'est pas l'esquive mais la visee. Le clavier tire sur huit rayons,
donc seuls 41 % des tirs pouvaient toucher (V37-V39), et les regles font
marcher le heros autour de sa cible pour l'aligner sur un rayon : il se
rapproche, tue deux fois moins vite, et se fait toucher par des ennemis
qu'il devrait tenir a distance sans effort.

Trois choses changent avant le premier essai.

1. **Les regles du jeu sont lues a la source.** Le jeu est public et son
   code l'est aussi (`js/data/enemies.js`, `js/game/boss.js`,
   `js/data/items.js`, `js/data/difficulty.js`, version 4.1.0 - celle que
   sert itch). Vitesses, rayons, duree de chaque telegraphe, effets de
   chaque relique : plus une constante du professeur n'est devinee. Le jeu
   joue toujours la version itch ; le code ne sert qu'a lire les regles.
2. **La visee passe par la souris.** `js/core/input.js` offre une visee a
   360 degres au clic gauche, entree officielle pour tout joueur. L'env la
   pilote en turbo (`step(action, aim)`), le clavier gardant la priorite
   comme dans le jeu. Les 96 actions et la baseline aleatoire a 14 actions
   ne bougent pas : le score, le budget et la baseline sont ceux des phases
   precedentes.
3. **La mesure devient appariee.** `SEED_SETUP_JS` remplace `Math.random`
   du jeu par un generateur seede : deux episodes de meme seed avec la
   meme politique sont identiques (verifie sur 971 steps, jusqu'a la mort).
   Les essais se jouent donc sur **80 donjons apparies** (`--seed 1000`),
   et `compare_paired.py` mesure la difference donjon par donjon.
   L'audit compte enfin le coup fatal (il s'arretait a l'etat `play`, donc
   un coup avant chaque mort) et imprime les **PV perdus par salle pooles**
   (somme des PV sur somme des salles) avec un intervalle a 95 % par
   bootstrap sur les episodes. Une politique est adoptee si cet intervalle
   exclut zero et si les salles par episode ne reculent pas.

Ce qui est appris et ce qui ne l'est pas sera nomme clairement : le reseau
apprend le pilotage (deplacement, dash, competence, interaction) ; la visee
et l'inventaire sont deux reflexes explicites, partages avec le professeur,
et le README donnera l'ablation reseau seul / agent complet / professeur.

### V45 - le point zero : 80 donjons apparies, le coup fatal compte

Le professeur de la phase 4 et le clone retenu (`bc_agent_v6`, dont les
trois features de valeur d'objet ajoutees apres son entrainement sont
neutralisees par des poids nuls) sont mesures sur les seeds 1000 a 1079.

| Politique | Score | Salles | Etages | PV / salle [IC 95 %] | Etage atteint 1:2:3:4:5 | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| Professeur ph. 4 | 358.81 +/- 23.85 | 978 | 112 | **1.032** [0.963, 1.108] | 8:45:17:7:3 | 2/80 |
| Clone v6 | 249.49 +/- 11.86 | 711 | 75 | **1.049** [0.996, 1.106] | 12:62:5:1:0 | 0/80 |

Avec le coup fatal, le professeur passe de 0.84 a 1.03 PV par salle : le
chiffre de la phase 4 sous-estimait chaque mort d'un demi-coeur a un coeur.
Ses 1 009 PV se repartissent ainsi : projectiles 325, Ame errante 113,
Vorace 102, eruptions 63, Grimoire dechaine 57 ; parmi les attaques de
gardien, `tripleLunge` 52, `heavyCharge` 47, `mirrorEcho` 37. Le clone
meurt plus tot (62 runs sur 80 s'arretent a l'etage 2) et perd ses PV aux
memes ennemis. Tout ce qui suit se compare a cette ligne.

### V46 - la visee a la souris

Une seule chose change : quand une regle tire, l'orbe part a la souris vers
l'ennemi le plus proche, au lieu du rayon clavier le plus proche de lui.
Aucune anticipation, aucun changement de cible ni de deplacement - les
regles marchent encore autour de leur cible pour « s'aligner », ce qui ne
sert plus a rien.

| Politique | Score | Salles / ep. | Etages / ep. | PV / salle [IC 95 %] | Etage atteint 1:2:3:4:5 | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| V45 professeur ph. 4 | 358.81 | 12.2 | 1.40 | 1.032 [0.963, 1.108] | 8:45:17:7:3 | 2/80 |
| **V46 souris** | **552.42** | **18.4** | **2.24** | **0.836 [0.758, 0.917]** | 2:17:32:18:11 | **7/80** |

Difference appariee sur les 80 donjons : +193.6 points (erreur type 37.9),
+6.2 salles, +0.84 etage par episode ; les PV par salle baissent de 0.196,
intervalle [-0.305, -0.093]. Le candidat marque plus sur 58 donjons sur 80.
**Adopte.** Les PV perdus par run montent pourtant (12.6 -> 15.4) parce que
les runs vont plus loin : 733 PV sont perdus aux etages 3 a 5 contre 263
avant, la ou le contact coute un coeur. La repartition par tueur change de
nature : les projectiles passent de 32 % a 36 % des PV, l'Ame errante tombe
de 113 a 85, et les gardiens profonds apparaissent (`magmaWake` 38,
`tidalPulse` 16, `sweepBeam` 11). Quatre runs se figent dans une salle
nettoyee (deux au point zero) : la salle est vide, les portes ouvertes, le
heros bouge sans sortir - un defaut de navigation, deja present, que la
duree de survie expose davantage.

### V47 a V49 - trois essais neutres, et un blocage enfin compris

Les trois essais suivants s'appuient sur le meme protocole (80 donjons
apparies, difference mesuree donjon par donjon) :

| Essai | Score | Salles / ep. | PV / salle [IC 95 %] | Victoires | Figees | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| V46 souris | 552.42 | 18.4 | 0.836 [0.758, 0.917] | 7/80 | 4 | reference |
| V47 zone morte 24 -> 6 px pres d'une cible | 541.87 | 18.0 | 0.861 [0.794, 0.934] | 6/80 | 5 | neutre (diff +0.025, IC [-0.073, +0.125]) |
| V48 anticipation du tir (lead amorti) | 559.13 | 18.6 | 0.832 [0.770, 0.892] | 5/80 | 4 | neutre (diff -0.029, IC [-0.115, +0.048]) |
| V49 priorite de cible | 515.06 | 17.2 | 0.864 [0.794, 0.938] | 5/80 | 1 | neutre (diff +0.032), -1.5 salle / ep. |

**V47** visait les runs figees de V46 : le heros tournait autour d'une
piece d'or a 36 px sans la ramasser. Reduire la zone morte de
`_move_from_delta` a repare les quatre seeds observes... et cinq autres
runs se sont figees ailleurs. La vraie cause est apparue en instrumentant
la salle : la piece est aimantee vers le heros a ~220 px/s au maximum
(`pickup.js`, acceleration 900 px/s2 amortie), et le heros court a
290 px/s. Il depasse la piece a chaque pas, elle reste 36 px derriere lui,
et le rayon de ramassage est de 28 px. S'arreter la ramasse en un
demi-seconde. La regle `_collect` attend donc l'aimant sous 100 px ; les
six seeds figes des trois audits jouent maintenant jusqu'au bout. Cette
correction est entree dans V50.

**V48**, l'anticipation du tir, resolue exactement cette fois (temps
d'interception d'un orbe a 580 px/s vers un point mobile, composante
laterale amortie selon la famille, plafond 0,6 s, aucune anticipation sur
un ennemi qui freine), ne change rien de mesurable. La visee a 360 degres
touchait deja ; le gain restant est dans le deplacement, pas dans l'angle.

**V49**, choisir la cible par ce qu'elle coute (invocateurs, bombes avant
armement, tireurs en preparation) plutot que par la distance, coute une
salle et demie par run : tirer sur un chuchoteur a 300 px laisse l'ame
errante a 100 px arriver au contact. La priorite reste disponible en
option (`teacher:priority`) mais n'est pas retenue telle quelle.

### V50 - le deplacement prédictif : la run se finit

Les regles de deplacement du clavier (`_best_direction` sur un champ de
danger, `_shot_dodge` sur le seul projectile le plus proche,
`_hazard_escape` radial) sont remplacees par une simulation
(`src/mover.py`) : les dix-sept deplacements que le clavier permet
(immobile, huit marches, huit dashs) sont joues sur 0,6 s a 20 Hz contre
ce que le jeu va faire dans ce laps de temps, tout etant lu dans son code -
orbes en vol, zones qui eclatent a la fin de leur meche, chargeur qui a
verrouille sa visee et va foncer sur cette droite (`enemy.js:329-336`),
bombe amorcee, tireur en preparation dont l'orbe partira a 0,52 s,
attaques de gardien annoncees (`boss.js` : eventail, anneaux, echo,
reflets, onde de choc, balayage, sillage, repliques). Un candidat touche
est elimine ; parmi les autres, le moins cher pour le combat gagne : bande
de distance par famille d'ennemi (`src/bestiary.py`, lue dans
`enemies.js` - un chargeur se tient a 240-350 px parce que sa charge fait
170-218 px, un tireur a 300-440, une bombe au-dela de sa distance
d'armement), ligne de vue, direction demandee, et un dash qui coute cinq
points quand une marche suffit (V27). La securite est une contrainte,
jamais un objectif : maximiser le degagement (V41-V42) envoyait le heros
dans les coins. Le tir part a la souris a chaque pas, dashs compris.

| Politique | Score | Salles / ep. | Etages / ep. | PV / salle [IC 95 %] | Etage atteint 3:4:5 | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| V49 | 515.06 | 17.2 | 2.17 | 0.864 [0.794, 0.938] | 41:16:7 | 5/80 |
| **V50 mover** | **992.23 +/- 15.87** | **32.2** | **3.84** | **0.211 [0.184, 0.239]** | 5:3:72 | **64/80** |

Difference appariee : +477 points (erreur type 29), +15.1 salles, +1.66
etage par episode, PV par salle -0.653, intervalle [-0.731, -0.576],
meilleur sur 75 donjons sur 80. **Adopte.** 72 runs sur 80 atteignent
l'etage 5, 64 le finissent ; les runs gagnantes perdent de 0 a 16 PV
(mediane 6). Le niveau d'environ 0,2 PV par salle, calcule en phase 4
comme celui qu'il faudrait pour finir regulierement, est atteint.

Ce qui reste, lu dans le meme audit : 8 morts, presque toutes au sillage
de magma (Pyroclaste 79 PV, Eclipse 55) et a la charge lourde de Vorace
(51) ; et **8 runs figees** aux etages 3 a 5, budget epuise sans mourir -
le prochain chantier. Les 80 PV attribues a « ? » ne sont pas des coups :
c'est l'autel du sacrifice, qui retire un coeur de maximum et fait baisser
la barre sans agresseur ; l'audit les ignore desormais (PV reels de V50
apres correction : ~0,18 par salle).

### V51 - les blocages, et un bug qui a failli passer pour un progres

Les 8 runs figees de V50 ont ete reproduites (le jeu est deterministe a
seed egale) et trois causes corrigees : un pilier entre le heros et la porte
que la regle clavier attaquait tout droit (desormais chaque deplacement hors
combat passe par la simulation, qui facture le trajet perdu contre un mur,
et la simulation laisse franchir les portes ouvertes) ; un combat contre
deux ennemis lents qui n'avancait plus a cause de tirs serpentants
(`sillage-sinueux` fait osciller l'orbe de +/-31 degres, `game.js:2189`) -
apres 600 pas dans la meme salle, la bande de distance se referme a 0-120 px
et un ennemi a un tir de la mort n'est plus lache ; et l'attente de la piece
d'or aimantee (V47).

Premiere mesure (V51) : **45 victoires au lieu de 64, 23 runs figees au lieu
de 8**. Le PV par salle baissait pourtant (0.170). Cause, trouvee en
affichant le cout de chaque candidat de la simulation : le candidat
« rester immobile » demandait 43 px de trajet et n'en obtenait aucun, donc
la simulation le comptait comme bloque par un mur (+5) et ne le choisissait
jamais. Le heros ne pouvait plus attendre une piece ni tenir une position ;
en V50, les deplacements hors combat contournaient la simulation, ce qui
masquait le defaut. Corrige en une ligne (`mover._simulate`).

| Essai | Score | Salles / ep. | PV / salle [IC 95 %] | Victoires | Figees |
| --- | --- | --- | --- | --- | --- |
| V50 mover | 992.23 | 32.2 | 0.211 [0.184, 0.239] | 64/80 | 7 |
| V51 corrections (bug) | 914.15 | 29.0 | 0.170 [0.142, 0.203] | 45/80 | 23 |
| **V51b corrections, immobilite reparee** | 985.67 | 32.0 | 0.209 [0.185, 0.235] | 60/80 | 6 |

V51b contre V50, apparie : PV par salle -0.002, intervalle [-0.041, +0.034],
victoires 64 -> 60. **Neutre.** Les corrections sont conservees parce
qu'elles reparent des cas reproduits, pas parce qu'elles gagnent quelque
chose de mesurable sur ces 80 donjons.

### V52 et V53 - la table des gardiens et l'economie de build : neutres

| Essai | Score | PV / salle [IC 95 %] | Victoires | Morts | Figees | Difference appariee (PV / salle) |
| --- | --- | --- | --- | --- | --- | --- |
| V51b | 985.67 | 0.209 [0.185, 0.235] | 60/80 | 13 | 6 | reference |
| V52 + table des gardiens | 963.20 | 0.181 [0.156, 0.209] | 59/80 | 15 | 7 | -0.027, IC [-0.057, +0.003] |
| V53 + economie de build | 1001.79 | 0.206 [0.176, 0.237] | 63/80 | 11 | 4 | +0.024, IC [-0.013, +0.062] |

**V52** (echo de la Tisseuse, reflets du Miroir, couronne de ronces, croix
de givre, triple bond sans dash, penalite pour finir un pas dans une zone
qui eclatera plus tard) fait perdre un peu moins de PV mais pas moins de
runs : le sillage de magma reste le premier tueur (Pyroclaste 99 PV,
Eclipse 48), et la table n'y change rien parce que la simulation le voyait
deja. **V53** (valeur d'objet sur tous les modificateurs, autel, parieur,
brasero, reserve d'or) gagne 4 victoires et perd 0.024 PV par salle : bruit
dans les deux sens. Ni l'une ni l'autre n'est retenue : le professeur gele
est **V51b**, `teacher:lead,priority,mover`. L'anticipation et la priorite
de cible (V48, V49) restent dans cette composition parce que le
deplacement predictif a ete mesure avec elles ; l'ablation qui les
retirerait n'a pas ete jouee, faute de temps.

### Validation sur 80 donjons jamais vus

Toutes les mesures precedentes tournent sur les seeds 1000 a 1079, et les
corrections de blocage ont ete concues en regardant ces donjons-la. Le
professeur gele est donc rejoue sur les seeds **2000 a 2079** :

| Donjons | Score | Salles / ep. | PV / salle [IC 95 %] | Etage 5 atteint | Victoires | Figees |
| --- | --- | --- | --- | --- | --- | --- |
| 1000-1079 (mesure) | 985.67 | 32.0 | 0.209 [0.185, 0.235] | 70/80 | 60/80 | 6 |
| 2000-2079 (jamais vus) | 1007.30 | 32.5 | 0.175 [0.150, 0.203] | 72/80 | 60/80 | 4 |

Meme niveau : les regles n'ont rien appris des donjons de mesure. Les
morts se ressemblent aussi (sillage de magma 80 PV, charge lourde 51,
sillage de l'Eclipse 46) : c'est la ou un futur professeur gagnerait.

### V54 - le clone de la phase 5

Le professeur gele joue 300 episodes (curriculum d'etages 1 1 2 3 4 5 4 5,
un donjon par episode, autres seeds que ceux de la mesure) : **1 203 849
paires** de 439 features, dont 63 713 aux etats que le reflexe
d'inventaire prend en charge et 454 800 dans une salle de gardien. Le
reseau a deux tetes (384x384, direction + modificateur), les paires sont
ponderees par classe (dash, competence, E sont 0,9 % des etiquettes) et
par etat (x3 chez un gardien, x2 aux etages 3+, x0,5 aux etats du
reflexe), avec lissage 0,05 et decroissance des poids 1e-4. Trente
epoques, une par 4 a 6 minutes, precision de validation **0.738** (0.658
a la premiere, plateau des l'epoque 16). C'est moins que les 0.82 de la
phase 4, sur une tache plus dure : le professeur choisit son deplacement
par une simulation que le reseau doit reconstituer depuis les features.

Evaluation en agent hybride (`audit_deaths.py --policy hybrid:`, epsilon
0) : le reseau pilote, les reflexes de visee et d'inventaire sont ceux du
professeur.

| Agent | Score | Salles / ep. | Etages / ep. | PV / salle [IC 95 %] | Etage 5 atteint | Victoires |
| --- | --- | --- | --- | --- | --- | --- |
| Clone phase 4 (`bc_agent_v6`, V45) | 249.49 | 8.9 | 0.94 | 1.049 [0.996, 1.106] | 0/80 | 0/80 |
| **Clone phase 5 (`bc2_agent`)** | **636.42 +/- 30.90** | **20.9** | **2.48** | **0.893 [0.835, 0.951]** | **20/80** | **7/80** |
| Professeur gele (V51b) | 985.67 | 32.0 | 3.77 | 0.209 [0.185, 0.235] | 70/80 | 60/80 |

Contre le clone de la phase 4 : 2,5 fois le score, 2,3 fois les salles,
et les **premieres victoires d'un agent appris** (7, dont une a 1 120
points). Contre son professeur, apparie : -349 points, -11 salles, PV par
salle +0.684 (intervalle [+0.622, +0.743]), 7 donjons sur 80 ou il fait
mieux. Le clone est a 65 % du professeur en score et en salles, mais a
quatre fois ses PV perdus par salle : il tient les etages 1-2 (11 morts),
meurt aux etages 3-4 (43 morts) et arrive vingt fois a l'etage 5. Ses
tueurs sont ceux du professeur de la phase 4 - les projectiles (35 % des
PV), la charge lourde et le triple bond de Vorace - c'est-a-dire
exactement ce que la simulation de deplacement esquive et que le reseau
n'a qu'imparfaitement appris. Neuf runs epuisent leur budget, dont une
figee dans une salle d'autel et une dans une boutique : des etats rares
dans les demonstrations.

Deux pistes, dans l'ordre : une ronde DAgger (le clone joue, le
professeur etiquette) pour couvrir ces etats-la, puis donner au reseau ce
que le professeur voit reellement quand il choisit son pas - pour chacun
des dix-sept deplacements, le verdict de la simulation sur le prochain
0,6 s - comme la phase 3 avait donne le champ de danger.

### V55 - une ronde DAgger, et l'ablation du reflexe d'inventaire

**DAgger.** Le clone joue 100 episodes (memes etages de depart que la
collecte, autres seeds), le professeur etiquette chaque etat : 168 925
paires de plus, sur les situations que le clone atteint reellement.
Re-entrainement a chaud depuis `bc2_agent` (8 epoques, pas 3e-4) sur les
deux jeux reunis : precision de validation 0.738 -> **0.760**.

| Agent | Score | Salles / ep. | PV / salle [IC 95 %] | Etage 5 | Victoires | Figees |
| --- | --- | --- | --- | --- | --- | --- |
| Clone v2 (V54) | 636.42 | 20.9 | 0.893 [0.835, 0.951] | 20/80 | 7/80 | 9 |
| **Clone v2 + DAgger (V55, `bc2_agent_d1`)** | **660.09** | **22.0** | **0.821 [0.764, 0.883]** | 20/80 | **10/80** | 5 |

Difference appariee : PV par salle -0.072, intervalle [-0.155, +0.008] -
il frole zero ; +24 points, +1.1 salle, +3 victoires, meilleur sur 42
donjons sur 80, et quatre runs figees de moins. Legerement positif,
**retenu** comme clone de reference, sans illusion : la ronde ne comble
pas l'ecart avec le professeur (0.82 PV par salle contre 0.21).

**Ablation.** Le meme reseau (V54) joue sans le reflexe d'inventaire : il
doit lui-meme marcher vers les piedestaux, presser E, acheter, prendre le
portail.

| Agent | Score | Salles / ep. | PV / salle | Etage 5 | Victoires |
| --- | --- | --- | --- | --- | --- |
| Clone v2 avec ses deux reflexes (V54) | 636.42 | 20.9 | 0.893 | 20/80 | 7/80 |
| Clone v2 sans reflexe d'inventaire | 430.10 | 14.7 | 0.650 | 0/80 | 0/80 |

Sans le reflexe, le reseau atteint encore l'etage 4 (11 runs) mais jamais
le 5 et ne gagne rien : il rate les reliques et arrive au duo de gardiens
avec un build trop faible. Ses PV par salle sont plus bas parce qu'il
meurt plus tot, la ou les salles coutent moins. Le reflexe d'inventaire
vaut donc environ 200 points et toutes les victoires ; la visee a la
souris n'a pas d'ablation propre (le reseau n'a pas de sortie de visee),
mais V45 -> V46 en donne l'ordre de grandeur sur le professeur.

**Ou est l'ecart.** Le clone reproduit 76 % des pas du professeur, et ce
sont les autres qui coutent : le professeur decide son deplacement par une
simulation des 0,6 s suivantes que le reseau doit reconstituer depuis les
positions et les vitesses. La phase 3 avait resolu un probleme du meme
genre en donnant au reseau le champ de danger plutot qu'en lui demandant
de le deviner. Prochain essai, dans le meme esprit : ajouter aux features
le verdict de la simulation pour chacun des dix-sept deplacements (touche
ou non, dans combien de temps, degagement), et re-cloner.

### V56 - donner au reseau ce que le professeur simule

Le professeur choisit son pas en simulant les 0,6 s suivantes pour chacun
des dix-sept deplacements que le clavier autorise (`src/mover.py`). Le
clone, lui, ne recevait que des positions et des vitesses : il devait
refaire cette simulation de tete. `src/features3.py` la lui donne - pour
chaque deplacement, quatre nombres : touche ou non, dans combien de temps,
quel degagement il garde, quelle distance a sa cible - soit **507
features** au lieu de 439. Le module de features est choisi par le champ
`feature_version` du modele, donc les clones v2 restent rechargeables.

Meme collecte que V54, aux memes seeds : les 1 203 849 memes paires, seul
le vecteur d'observation change. Vingt-deux epoques, precision de
validation 0.738 -> **0.772**.

| Agent | Score | Salles / ep. | PV / salle [IC 95 %] | Etage 5 | Victoires | Figees |
| --- | --- | --- | --- | --- | --- | --- |
| Clone v2 + DAgger (V55) | 660.09 | 22.0 | 0.821 [0.764, 0.883] | 20/80 | 10/80 | 5 |
| Clone v3 (V56, `bc3_agent`) | 740.17 | 23.0 | **0.254 [0.219, 0.290]** | 39/80 | 35/80 | **38** |

C'est le plus gros effet mesure depuis le debut de la phase : les PV
perdus par salle passent de 0.821 a 0.254, difference appariee -0.567,
intervalle [-0.635, -0.501]. Le clone ne prend presque plus de coups, et
gagne trente-cinq donjons contre dix.

Mais **trente-huit runs sur quatre-vingts restent bloquees** plus de 1 500
pas sans nettoyer une salle, trente-six epuisent leur budget sans mourir -
six d'entre elles dans la salle de depart de l'etage 1 - et la mediane du
plus long blocage passe de 450 a 1 386 pas. Le verdict de la simulation
rend le reseau prudent au point de l'immobiliser : quand aucun deplacement
n'est sur, il n'en prend aucun. Une bonne part des 740 points est du temps
de survie paye (`time * 0.1`), la pathologie deja rencontree en V10, V30
et V34. Le gain est reel, mais l'essai ne s'adopte pas tel quel.

### V57 - la ronde DAgger qui remet le clone en marche

Le clone v3 joue 100 episodes, le professeur etiquette chaque etat :
**372 815 paires**, deux fois plus qu'a la ronde precedente - un clone qui
s'attarde produit plus de pas. Re-entrainement a chaud depuis `bc3_agent`,
huit epoques a 3e-4 sur les deux jeux reunis : precision de validation
0.772 -> **0.766**. Elle baisse pendant que le jeu s'ameliore, ce qui est
attendu : les etats ajoutes sont ceux ou le clone s'enlisait.

| Agent | Score | Salles / ep. | Etages / ep. | PV / salle [IC 95 %] | Etage 5 | Victoires | Figees |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clone v2 + DAgger (V55) | 660.09 | 22.0 | 2.61 | 0.821 [0.764, 0.883] | 20/80 | 10/80 | 5 |
| Clone v3 (V56) | 740.17 | 23.0 | 2.65 | 0.254 [0.219, 0.290] | 39/80 | 35/80 | 38 |
| **Clone v3 + DAgger (V57, `bc3_agent_d1`)** | **921.36 +/- 24.62** | **29.5** | **3.52** | **0.292 [0.261, 0.329]** | **58/80** | **49/80** | **12** |
| Professeur gele (V51b) | 985.67 | 32.0 | 3.77 | 0.209 [0.185, 0.235] | 70/80 | 60/80 | 6 |

Difference appariee V56 -> V57 : +181.2 points (err. 45.3), +6.5 salles,
+0.88 etage, +45 kills, toutes reelles. Les blocages tombent de 38 a 12 et
les PV par salle ne remontent qu'a peine - 0.254 -> 0.292, intervalle
[-0.010, +0.090], neutre : le reseau ne redevient pas imprudent, il se
remet en marche. **Quarante-neuf victoires sur quatre-vingts.**

Face au professeur gele, sur les memes donjons : -64.3 points, -2.5
salles, -0.25 etage, +0.083 PV par salle [+0.042, +0.126], et le clone
marque plus haut sur 29 donjons sur 80. Il est a **93 % de son professeur
en score, 92 % en salles, 82 % de ses victoires**. Le plafond de 76 % que
ni deux rondes DAgger ni un reseau 512x512 n'avaient su lever tenait donc
a une information absente du vecteur d'observation, pas a la capacite du
reseau ni au nombre de demonstrations. C'est la meme lecon qu'a la phase 3
avec le champ de danger, et elle a mis deux phases a etre reapprise.

**Retenu : `models/bc3_agent_d1.json`**, clone de reference du projet.

Ce qui reste. Dix-sept runs epuisent leur budget : huit en salle de combat,
quatre en boutique - le clone s'attarde devant l'etal, un etat rare dans
les demonstrations. Les projectiles font toujours 34 % des PV perdus,
l'eruption tellurique 20 %. Et les PV se perdent desormais aux etages 3 a 5
(278 des 396), c'est-a-dire la ou l'agent va maintenant.

## Limites connues

- Le score de comparaison paie le temps de survie (`time * 0.1`). Il est
  conserve tel quel - le changer obligerait a re-mesurer tout l'historique -
  mais il ne suffit pas a lui seul a classer deux politiques : salles, etages
  et blocages accompagnent desormais chaque chiffre. V56 en est le dernier
  rappel.
- Les donjons n'ont ete rendus reproductibles qu'a la phase 5. Les essais des
  phases 1 a 4 se sont donc decides sur 20 a 40 episodes non apparies, ou le
  meme modele mesure deux fois donnait 278.07 puis 251.57 : ce bruit est dans
  tous les chiffres anterieurs a V45.
- Les modeles des phases 2 a 4 ne sont plus rechargeables : l'observation est
  passee de 109 a 202, 212, 439 puis 507 dimensions, et l'espace d'actions de
  59 a 64 puis 96. Leurs resultats restent au carnet ; la demonstration
  rejouable est celle de la phase 5.
- L'agent est hierarchique et l'annonce : la visee et l'inventaire sont des
  reflexes ecrits, partages avec le professeur, et c'est le pilotage qui est
  appris. L'ablation de V55 chiffre le second - environ 200 points et toutes
  les victoires.
- L'attrition reste le mur, cote professeur : 0.209 PV par salle, soit une
  dizaine de PV sur une run complete de 46 salles. La marge est mince, et ses
  vingt echecs sur quatre-vingts sont la.
- Le clone reste a 93 % de son professeur en score et a 82 % de ses victoires.
  L'ecart tenait a une information absente des features jusqu'a V56 ; ce qui
  en reste n'a pas ete explique.
- Les evaluations sont conservees dans `reports/` - `eval_bc_v6.csv` pour la
  phase 4, `death_audit_clone3_v57_dagger*.csv` pour le clone retenu - pour
  relier chaque chiffre a son fichier source.

## Pistes avec plus de temps

- Rendre les donjons reproductibles (remplacer `game.rng` par un generateur
  seede a l'ouverture de la page) pour faire des comparaisons **appariees** :
  c'est la seule facon de sortir du bruit sans multiplier les episodes.
- Reduire encore le cout par salle : l'audit de la phase 4 montre que les
  coups restants se prennent en majorite au corps a corps, avec un ou deux
  ennemis, dans des salles de combat ordinaires.
- Utiliser la competence du heros (anneau de 12 eclats toutes les 7 s), encore
  presque inexploitee, et evaluer l'autel du sacrifice, systematiquement
  refuse aujourd'hui.
- Reprendre une passe RL depuis le clone de la phase 4, avec une recompense
  qui paie la salle et l'etage plutot que la survie (voir V34).
