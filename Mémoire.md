# Introduction

## L'effet cigogne dans le cas de l'apprentissage automatique par renforcement

### Problématique

- Peut-on utiliser l'apprentissage par renforcement pour différencier ces corrélations de causalités ?

- Dans quelle mesure ce biais influence-t-il le résultat de prédictions basées sur l'apprentissage automatique ?

### Réflexion

Tout vient d'une réflexion sur la différence entre corrélation et causalité. C'est un problème qui survient systématiquement lorsqu'on fait des statistiques descriptives à partir de données dont on dispose. 

L'apprentissage automatique peut lui aussi subir ce biais, ne serait-ce que parce que le biais vient des données en entrées (malgré une validation croisée).

Comment peut-on répondre à ce biais ?

Avec une randomisation : couper aléatoirement un échantillon en deux, agir sur seulement l'une des moitiés, comparer les résultats.

Comment peut-on répondre à ce biais automatiquement ?

On peut imaginer un algorithme qui va apprendre non pas à partir de données en entrée, mais face à une situation, en lui permettant d'agir dessus suivant un protocole expérimental rigoureux et robuste face à la confusion corrélation/causalité via un apprentissage par renforcement.


# I- Revue de littérature

## A- Une question triviale ?

### a) Dans le cas d'une régression

Dans le cas d'une régression, la question peut sembler triviale.

Puisqu'une régression consiste à mesurer les corrélations entre toutes les variables afin d'en estimer une à partir des autres, il suffit que certaines variables soient corrélées avec la celle à estimer pour biaiser les résultats. Pour combattre ce biais, il faut qu'un être humain analyse le contexte pour déterminer s'il y a causalité entre les variables.

### b) Dans le cas de l'apprentissage par renforcement

Si le cas de la régression semblait trivial, c'est peut-être parce que l'algorithme n'a pas l'occasion d'interagir avec son environnement pour tester ce qui est une corrélation et ce qui est une causalité.

On peut d'ailleurs considérer trois types de variables : 

- les données intrinsèques à l'environnement

- les entrées (les actions de l'agent sur l'environnement)

- Les sorties issues des actions sur l'environnement (la récompense)


## B- Sous-catégories d'effet cigogne

### a) Données d'apprentissage différentes non représentatives

- https://app.wandb.ai/stacey/aprl/reports/Adversarial-Policies-in-Multi-Agent-Settings--VmlldzoxMDEyNzE

Résumé du protocole de cette publication :

On prend deux agents, A et B, et un jeu compétitif.

A apprend à jouer à partir de données de véritables joueurs.

Puis B apprend à jouer contre A

Il en résulte que la meilleure manière pour B de gagner consiste à ne pas jouer.

En effet, A n'a appris à jouer que contre des personnes qui savent jouer. B faisant des choses inattendues, A perd tout seul.


# II- Expérimentation

Faire interagir un agent suivant plusieurs algorithmes d'apprentissage par renforcement pour apprendre face à un environnement biaisé.

Pour créer des situation biaisées, on préfèrera utiliser une librairie permettant de créer un environnement.

J'ai choisi (/ commencé à utiliser) Tensorflow, car la documentation semble claire et bien fournie, et que la librairie implémente plusieurs algorithmes.

## A- L'agent et les algorithmes

Algorithmes présents dans TF :

- [DQN][1]

- [REINFORCE][2]

- [DDPG][3]

- [TD3][4]

- [PPO][5]

- [SAC][6]

TODO: réussir à les faire marcher, puis écrire une boucle for qui enregistre les résultats de chaque algos pour une configuration de l'environnement donnée


## B- L'environnement

On peut imaginer des prix mis à jour en temps réel par l'agent, l'objectif de l'agent étant de trouver le prix maximisant le résultat net.
L'environnement répondrait, pour chaque offre, une demande (un nombre d'achats).

Cas réels qui correspondraient : prix dans un centre commercial connecté, sur un site d'e-commerce, sur un marché à terme en temps réel (financier, de l'électricité, du blé), etc.

### a) L'implémentation

Dans TF, on peut créer deux types d'environnement : py_environment.PyEnvironment ou tf_environment.TFEnvironment 

- Temps discret

- Action: pour chaque lieu et/ou produit, un prix de vente

- Observation: pour chaque lieu et/ou produit, une demande

- Récompense: la somme, pour chaque lieu et/ou produit, du prix de vente auquel on soustrait le prix d'achat.

### b) Paramètres à ajouter

- élasticité demande selon offre

- prix plafond et prix plancher

- un seul lieu d'achat/vente ou bien plusieurs avec des prix différents

- les caractéristiques des lieux d'échange (nombre de clients, intérêt pour le produit, prix plafond et plancher)

- un seul ou plusieurs produits

- produits qui répondent ou non au même besoin

- coûts des produits identiques ou non

- coûts des produits qui changent au fil du temps ou non

- quelle part d'aléatoire

## C- Les biais à implémenter

A une étape de l'algorithme, arbitrairement jouter ou supprimer des magasins ou produits avec des caractéristiques non représentatives de la population de départ.

Exemples : 

- l'expérience était sur les carrefour city, elle inclut par la suite également les carrefour market, d'une taille en moyenne différente. Toutes les quantités varient.

- la chaîne s'étend sur un territoire avec des habitudes de consommation différentes


# III-  Analyse des résultats

Questions auxquelles on souhaitait répondre, (1) :

- Quelle est l'efficacité l'agent qui ne peut apprendre qu'avec des données ayant déjà été observées (2) (apprentissage supervisé) 

- Versus l'efficacité de l'apprentissage par renforcement seul avec randomisation Versus l'efficacité de l'apprentissage par renforcement seul sans randomisation 

- Versus l'efficacité si on pré-entraine l'agent avec des données ayant déjà été observées puis qu'on le laisse se renforcer sans randomisation (apprentissage supervisé et par renforcement) 

- Versus l'efficacité si on pré-entraine l'agent avec des données ayant déjà été observées puis qu'on le laisse se renforcer avec randomisation (apprentissage supervisé et par renforcement)

## A- Si la question est triviale

## B- Si la question ne l'est pas: 

## C- Biais

## D- Ouverture

# Conclusion


# Sources

[1]: https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

[2]: http://www-anw.cs.umass.edu/%7Ebarto/courses/cs687/williams92simple.pdf

[3]: https://arxiv.org/pdf/1509.02971.pdf

[4]: https://arxiv.org/pdf/1802.09477.pdf

[5]: https://arxiv.org/abs/1707.06347

[6]: https://arxiv.org/abs/1801.01290