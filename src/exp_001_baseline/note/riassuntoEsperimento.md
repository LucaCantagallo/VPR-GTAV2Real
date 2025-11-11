# Esperimento 001 — Baseline (GTA → Tokyo247)

## Descrizione generale

Questo esperimento ha lo scopo di stabilire una **baseline di riferimento** per il problema di *Visual Place Recognition (VPR)*, addestrando un modello su un dataset sintetico (GTA V) e testandolo su un dataset reale (Tokyo247).  
L’obiettivo è valutare l’efficacia del codice sviluppato originariamente da **Matteo Scucchia**, dottorando che ha realizzato la prima versione della pipeline.  
Il mio contributo è consistito nel **rieseguire e validare** tale pipeline, apportando alcune **integrazioni funzionali e di analisi** per ottenere una baseline robusta e documentata.

---

## Modifiche e aggiunte personali

Rispetto al codice originale di Scucchia, sono state introdotte le seguenti migliorie:

- **Sistema di notifica Telegram** per il monitoraggio automatico del completamento delle fasi di training, testing e calcolo delle metriche.  
- **Metriche aggiuntive**, tra cui:
  - *Average Precision (AP)*
  - *Similarity Distribution*
  - Plot e visualizzazioni delle metriche principali.  
- Ottimizzazione minore del caricamento e salvataggio dei risultati per esperimenti multipli.  

---

## Setup dell’esperimento

- **Compito**: VPR (GTA→Tokyo247)  
- **Modello base**: ResNet50 con testa MLP Cosine  
- **Loss**: TripletMarginWithDistanceLoss (distanza = 1 - cosine similarity)  
- **Dataset di training**: GTA V (giorno ↔ notte)  
- **Dataset di test**: Tokyo247 (giorno ↔ notte)  
- **Numero di epoche testate**: 1 e 400  
- **Pretraining**: ImageNet e Places365  
- **Parametri trainabili**: 
  - `trainable_from_layer = null` → solo MLP addestrabile  
  - `trainable_from_layer = all` → rete intera addestrabile  

---

## Risultati (valid_size = 25)

| Pretraining | Trainable from | Epoche | Recall@1 | Recall@5 | Recall@10 | Recall@50 |
|--------------|----------------|---------|-----------|-----------|------------|------------|
| **ImageNet** | `null` | 1 | 0.467 | 0.637 | 0.723 | 0.904 |
| **ImageNet** | `null` | 400 | 0.488 | 0.685 | 0.747 | 0.901 |
| **ImageNet** | `all` | 1 | 0.344 | 0.515 | 0.619 | 0.803 |
| **ImageNet** | `all` | 400 | 0.147 | 0.328 | 0.419 | 0.699 |
| **Places365** | `null` | 1 | 0.400 | 0.571 | 0.672 | 0.848 |
| **Places365** | `null` | 400 | 0.181 | 0.339 | 0.448 | 0.723 |
| **Places365** | `all` | 1 | 0.355 | 0.581 | 0.664 | 0.851 |
| **Places365** | `all` | 400 | 0.088 | 0.195 | 0.259 | 0.501 |

---

## Analisi dei risultati

### Tendenze principali

1. **Congelare i pesi di base (trainable_from_layer = null)** produce risultati **migliori e più stabili** rispetto all’addestramento completo.  
2. Il **pretraining su ImageNet** offre prestazioni generalmente **superiori** rispetto a Places365, in particolare per i top-k più bassi.  
3. L’aumento del numero di epoche **non sempre migliora** le performance: in diversi casi, un training prolungato causa un **overfitting** sul dominio sintetico, peggiorando il trasferimento verso immagini reali.  
4. Il setup **ImageNet + trainable_from_layer=null + 400 epoche** risulta la combinazione più efficace.  

---

## Combinazione più performante

| Configurazione | Recall@1 | Recall@5 | Recall@10 | Recall@50 |
|----------------|-----------|-----------|------------|------------|
| **ImageNet + trainable_from_layer = null + 400 epoche** | **0.488** | **0.685** | **0.747** | **0.901** |

Questa combinazione è stata utilizzata come **baseline di riferimento** per gli esperimenti futuri.

---

## Esperimento aggiuntivo (valid_size = 0)

È stato condotto un test supplementare utilizzando la combinazione più performante (**ImageNet + null + 400 epoche**) ma senza set di validazione (valid_size = 0).  
I risultati mostrano un **ulteriore miglioramento** generale:

| k | Recall |
|---|--------|
| 1 | 0.485 |
| 5 | 0.669 |
| 10 | 0.749 |
| 50 | 0.923 |

Questo suggerisce che **tutti i dati disponibili contribuiscono positivamente** al training, riducendo la varianza e migliorando la generalizzazione.

---

## Conclusioni

- La pipeline di Scucchia è **replicabile e funzionale**, e costituisce un buon punto di partenza per esperimenti futuri.  
- Il miglior compromesso tra stabilità e performance si ottiene **con pesi ImageNet, rete parzialmente congelata (solo MLP addestrabile)** e un numero moderato di epoche.  
- Il test con `valid_size = 0` conferma la solidità di questa configurazione e stabilisce la **baseline ufficiale** per il mio lavoro successivo su metriche avanzate e tecniche di adattamento dominio-reale.  

---

## Sviluppi futuri proposti per l’Esperimento 2

L’Esperimento 2 si propone come un’estensione leggera dell’attuale configurazione, mirata a una migliore stabilità del training e a una riduzione del rumore nella convergenza, mantenendo invariata la struttura complessiva del pipeline.  

### Ottimizzazioni e modifiche proposte

1. **Early Stopping**  
   Implementare una strategia di arresto anticipato basata sulla *valid loss*, con soglia di pazienza di 5–10 epoche, per prevenire overfitting e ridurre oscillazioni inutili.

2. **Riduzione del Learning Rate (LR Scheduling)**  
   Integrare un `ReduceLROnPlateau` o un `CosineAnnealingLR` per diminuire progressivamente il learning rate in caso di stagnazione della valid loss, favorendo una convergenza più stabile.

3. **Weight Decay e Regularizzazione Leggera**  
   Aggiungere un lieve *weight decay* all’ottimizzatore Adam per contenere la crescita dei pesi e migliorare la generalizzazione senza alterare significativamente la dinamica di training.

4. **Monitoraggio Dettagliato delle Distanze Cosine**  
   Loggare durante il training la media delle distanze *anchor–positive* e *anchor–negative*, utile per verificare che la loss evolva coerentemente con la separazione semantica appresa.

### Obiettivo
Queste modifiche puntano a migliorare la stabilità e la riproducibilità del training, mantenendo il setup snello dell’esperimento 1 ma con una maggiore robustezza numerica, preparandolo alla futura estensione del dominio reale nell’Esperimento 3.

---

*Autore della baseline originale: Matteo Scucchia*  
*Riesecuzione, estensioni e analisi: Luca Cantagallo*  
*Anno: 2025*
