# Deuxième campagne de performances — 19 octobre 2026

Aucune nouvelle optimisation n'est activée à l'issue de cette campagne. Les essais n'ont pas démontré de gain assez régulier en génération pour remplacer le moteur existant. Le serveur et ses trois bibliothèques ont été restaurés à l'identique depuis les copies réalisées au début de cette campagne ; leurs SHA-256 ont été comparés. LTO et PGO sont désactivés, les n-grammes ne sont pas ajoutés au lancement de l'application et les paramètres enregistrés restent inchangés.

La vectorisation Hadamard AVX-512 de la [première campagne](performance-2026-10-19.md) fait partie de la référence de départ et reste présente. Le gain supplémentaire retenu dans cette deuxième campagne est donc **0 %**.

## Résultats

Débits médians de génération, en tokens/s. Chaque ligne compare une variante à son propre témoin mesuré en alternance pendant la même série. Les différences de débit entre séries reflètent aussi les variations de l'état de la machine : elles ne constituent pas des gains entre lignes.

| Essai | Modèle | Témoin | Variante | Écart des médianes | Paires mesurées | Décision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LTO, confirmation | Bonsai 2B PQ2 | 47,57 | 47,61 | +0,09 % | 8 | Désactivé : aucun gain régulier |
| PGO ciblé | Bonsai 2B PQ2 | 48,77 | 48,91 | +0,29 % | 8 | Désactivé : aucun gain régulier |
| LTO | Qwen3.5 2B Q5_K_M | 18,99 | 18,94 | −0,23 % | 6 | Désactivé |
| PGO ciblé | Qwen3.5 2B Q5_K_M | 18,99 | 18,94 | −0,24 % | 6 | Désactivé |
| Fusion multiplication/Hadamard | Bonsai 2B PQ2 | 47,75 | 47,64 | −0,24 % | 4 | Code retiré |
| Noyau PQ2 à 32 lignes, confirmation | Bonsai 2B PQ2 | 49,09 | 48,64 | −0,90 % | 8 | Code retiré |
| N-grammes à clé de 8 tokens, rédaction libre | Bonsai 2B PQ2 | 47,41 | 47,16 | −0,54 % | 5 | Pas d'activation globale |
| Même réglage, copie répétitive | Bonsai 2B PQ2 | 48,72 | 51,02 | +4,73 % | 5 | Gain conditionnel ; pas d'activation globale |

Les témoins de la fusion et du noyau à 32 lignes utilisent la copie LTO afin d'isoler l'effet de chaque modification. Les autres lignes utilisent le moteur initial de cette campagne. Pour les deux dernières lignes, la température est nulle et les réponses avant/après ont des empreintes identiques.

Le petit écart positif de PGO sur Bonsai ne constitue pas un gain confirmé : la médiane des huit ratios appariés est de −0,09 %, avec une paire à +4,91 %. Sur Qwen, cinq des six paires PGO sont négatives. Les écarts de quelques dixièmes de pourcent ne sont pas présentés comme des régressions établies ; ils suffisent ici à ne pas retenir une option qui devait améliorer le débit.

## Ce qui a été essayé

- **LTO** : compilation avec `GGML_LTO=ON`, sans changer les calculs ni la quantification. Le premier comparatif Bonsai indiquait +0,47 % sur quatre paires ; les huit paires de confirmation ne donnent plus que +0,09 %. Le contrôle Qwen ne montre pas de bénéfice non plus.
- **Fusion multiplication/Hadamard** : le backend CPU reconnaissait une multiplication F32, suivie d'un reshape puis d'un Hadamard, et calculait les deux opérations dans une seule étape. Les produits intermédiaires restaient disponibles pour les autres consommateurs du graphe. Les tests de graphe à 1/4/7 threads passaient avec égalité exacte. Le traitement du prompt gagnait environ 3,1 % dans ce premier essai, mais le débit de génération ne progressait pas : modification retirée.
- **PQ2 à 32 lignes** : deux groupes AVX-512 partageaient les mêmes broadcasts d'activations, avec les anciens noyaux à 16 et 8 lignes pour les restes. Les tests incluaient plusieurs largeurs de matrice et les groupes incomplets. Le premier comparatif indiquait +1,63 %, mais les huit paires de confirmation donnent −0,90 % ; ce résultat illustre pourquoi un premier gain apparent ne suffit pas. Modification retirée.
- **PGO ciblé** : instrumentation de `iqk_gemm_1bit.cpp` et `iqk_cpu_ops.cpp`, collecte de profils sur Bonsai 2B, Bonsai 4B et Qwen 2B, puis compilation avec `-fprofile-use` et `-fprofile-correction`. Les compteurs étaient atomiques ; après un premier essai trop coûteux à quatre threads, la collecte complète utilisait un seul thread, des prompts de 32/128 tokens et des générations de 8 tokens. Les mesures finales utilisaient quatre threads et 128 tokens, sans compteurs d'instrumentation. Aucun avertissement de profil manquant ou incohérent n'a été observé lors de cette compilation. Option retirée après les comparaisons.
- **N-grammes** : essais de `ngram-simple` avec un maximum de 8 tokens proposés et des clés de 4 puis 8 tokens. Le réglage à clé de 4 donne environ +9,65 % sur une copie au format conversation, mais −0,94 % en rédaction libre. Le réglage à clé de 8 reste conditionnel, comme le montre le tableau. Un texte artificiel très répétitif atteint même +230 %, avec EOS ignoré : ce cas favorable ne représente pas un gain général dans l'application. Aucun de ces réglages n'est activé par défaut.

Bonsai 4B a servi à l'apprentissage PGO ; cette campagne ne revendique pas de gain de génération mesuré sur ce modèle.

## Méthode et vérifications

Machine : Intel Core i7-1195G7, quatre cœurs physiques, huit processeurs logiques. Compilation Release native GCC 13.3.0 et OpenMP. Les comparaisons utilisent quatre threads de génération et de traitement du prompt, un contexte de 4 096 tokens, le repacking existant et les réglages d'attention existants.

Les serveurs comparés restent chargés simultanément. Une seule requête calcule à la fois ; l'ordre est inversé à chaque tour. Une génération de chauffe par variante et scénario est exclue. Chaque requête de comparaison produit exactement 128 tokens, avec seed 42, sans réutilisation du prompt et avec EOS ignoré. Les débits proviennent des timings du moteur, hors chargement et recherche web. La température est de 0,5, sauf pour la confirmation des n-grammes à clé de 8.

Les comparatifs de compilation, fusion et noyau sur Bonsai utilisent le même prompt synthétique de 64 tokens. Le comparatif Qwen utilise un vrai template de conversation de 52 tokens, avec `enable_thinking=false`, comme l'application. Les essais de n-grammes comprennent des prompts synthétiques et quatre conversations : explication, récit, code et copie répétitive. Leur résultat dépend du contenu ; les sorties avec échantillonnage ne sont pas toutes identiques avec et sans spéculation.

Au total : **181 requêtes de comparaison, dont 150 hors chauffe**, et 14 requêtes de collecte PGO achevées et enregistrées, dont deux dans le premier essai interrompu. Les débits instrumentés ne participent à aucun gain annoncé. Les comparatifs natifs de compilation et des noyaux produisent des réponses d'empreintes identiques entre les variantes comparées.

Vérifications effectuées :

- `test-tied-output` et `test-cpu-buffer` passent pour les variantes natives testées et après restauration du moteur initial.
- Les tests temporaires de fusion et de noyau étendu passent avant leur retrait avec les modifications correspondantes.
- Les 66 tests Python de l'application passent ; aucun code de l'application n'est modifié.
- Le script de mesure est syntaxiquement valide et ses nouveaux chemins de prompts et de templates sont exercés par les requêtes réelles.
- Les quatre fichiers binaires restaurés ont les mêmes SHA-256 que les copies initiales. Les options de génération de profils, d'utilisation des profils et de LTO sont absentes des commandes de compilation finales.

Seuls le script de benchmark et la documentation évoluent dans les fichiers suivis du dépôt. Les modifications natives expérimentales sont intégralement retirées.

## Données et reproduction

Les données locales sont conservées dans `.local-chat/performance-round2/`, ignoré par Git :

- `summary.json` : médianes, ratios appariés, identités des sorties et compteurs de spéculation ;
- `*.jsonl` et `*-config.json` : requêtes mesurées et configurations ;
- `binaries/{baseline,lto,fusion,pq2-32row,pgo}/` : serveur et bibliothèques de chaque variante ;
- `binary-sha256.json` et `restoration.json` : empreintes et contrôle de restauration ;
- journaux de compilation, tests, profils PGO et copies des sources expérimentales.

Par exemple, pour refaire la comparaison Qwen sans modifier les réglages de l'application :

```sh
python3 scripts/benchmark-local.py \
  .local-chat/performance-round2/qwen-compare-config.json \
  .local-chat/performance-round2/qwen-rerun.jsonl
```

Le fichier de sortie doit être nouveau. Les configurations de comparaison utilisent les copies archivées des bibliothèques afin de ne pas dépendre du contenu courant de `build/`.
