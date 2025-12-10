# Esperimento 002 — Ottimizzazioni, Generalizzazione VPR e Contrastive Learning

## Descrizione generale

Questo esperimento rappresenta l’evoluzione naturale dell’Esperimento 001.  
L’obiettivo è stato di **migliorare l’efficienza del training**, **generalizzare la pipeline** oltre il compito specifico giorno-notte e introdurre tecniche moderne di **contrastive learning**, mantenendo come riferimento il problema VPR e consolidando progressivamente una baseline più robusta.

Il lavoro si è sviluppato in più fasi consecutive e complementari:  
1. Ottimizzazione del training tramite early stopping e scheduling del learning rate.  
2. Refactor della pipeline per supportare compiti generici VPR oltre il giorno-notte.  
3. Estensione delle capacità del dataloader (multi-triplette per place).  
4. Migliorie incrementali con normalizzazione, scelta dell’optimizer e tuning dei parametri.  
5. Introduzione e integrazione completa del metodo **InfoNCE** come alternativa a Triplet Loss.  
6. Studio e valutazione degli hard negative nelle triplet.

---
## 1. Miglioramento dell’efficienza del training

Nell’Esperimento 1 il modello migliore richiedeva **400 epoche**.  
Con l’introduzione dei parametri configurabili nel blocco `train:` del YAML, la pipeline supporta ora:

- `early_stopping = true`  
- `patience = 10`  
- `min_delta = 0.002`  
- `reduce_lr_on_plateau = true`  
- `lr_patience = 3`  
- `lr_factor = 0.5`  
- `lr_min = 1e-6`  
- `weight_decay = 1e-6`  
- `optimizer = adam`

il training converge ora in **~25 epoche**, riducendo drasticamente i tempi computazionali e stabilizzando la curva di convergenza.



## 2. Refactor strutturale della pipeline (generalizzazione VPR)

La pipeline originale era vincolata al task **daynight**.  
Con il refactor è stata introdotta una struttura generalizzata, selezionabile tramite parametro YAML:

- `task = vpr`  
- `task = daynight`

Gli aggiornamenti principali:

### Generalizzazione dataset

È stato introdotto `places_extractor.py`, che:

- astrae completamente il recupero dei “places”;  
- supporta **GTA**, **Alderley**, **Tokyo247**, **GSV**;  
- rimuove logiche legate al giorno/notte;  
- introduce il parametro `percentage` per campionare sottoinsiemi controllati;  
- gestisce l'uso dello stesso dataset per più compiti (es. validation e testing), splittandolo in sottoinsiemi che mantengono la stessa varietà e garantiscono zero intersezioni di 'places'
- adotta grouping coerente per ciascun dataset (ad es. grouping per place-id in GSV).

Questa separazione rende il flusso indipendente dal tipo di dataset di training, validation o testing.

### Flusso daynight → flusso vpr

Il vecchio file `data_loader_daynight.py` continua a gestire:

- accoppiamento giorno/notte;  
- selezione di coppie basate sugli specifici pattern di GTA, Alderley, Tokyo247.

Per VPR è stato introdotto `filter_paired_vpr.py`, che:

- genera coppie generiche senza presupporre condizioni di illuminazione;  
- lavora uniformemente su tutti i dataset;  
- consente valutazione VPR basata su simple-pair matching.

### Standardizzazione della pipeline

Il flusso training/validation/testing ora è unificato:

- struttura di dataset completamente modulare;  
- nessuna logica special-case nei file principali;  
- configurazione interamente demandata ai parametri YAML.

### Nuovo setup VPR

Il refactor introduce una configurazione standardizzata:

- **train_dataset**: GTA  
- **validation_dataset**: GSV (subset indipendente dal testing set)  
- **test_dataset**: GSV (subset indipendente dal validation set)

Questo setup permette una baseline VPR completamente generalizzabile.  

Oltre alla generalizzazione, il refactor garantisce ora:

- directory di lavoro coerenti (`get_train_work_dir()` e `get_test_work_dir()`);  
- separazione chiara tra logica modello (`init_model`) e logica esperimento;  
- codice pronto per estendere futuri task senza modificare il core.

### Risultati baseline VPR

L’esperimento di riferimento, eseguito con la pipeline refactorizzata e con gli stessi parametri dell'esperimento precedente ha prodotto i seguenti valori di **recall@k**:

| k  | Recall |
|----|---------|
| 1  | 0.4011  |
| 5  | 0.5987  |
| **10** | **0.6402**  |
| 50 | 0.8147  |

Questi costituiscono la **baseline ufficiale** del nuovo flusso VPR.

## 3. Estensione e completa generalizzazione della generazione delle triplette

Il sistema precedente era rigidamente strutturato:  
- una sola tripletta per place,  
- generazione delle triplette **sparsa in più file**,  
- logica del sampling **fusa dentro train.py**,  
- impossibilità di usare approcci diversi dalla Triplet Loss senza riscrivere tutto.

Il refactor ha isolato completamente il meccanismo delle triplette all’interno di un modulo dedicato, rendendolo indipendente sia dal dataset che dal metodo di apprendimento.

### Architettura attuale

L’intera logica è ora contenuta in `triplet_loader.py`, che fornisce:

- **`get_dataloaders()`**  
  costruisce i DataLoader per train/validation in modo uniforme per tutti i dataset e tutti i metodi.
  
- **`refresh_dataloaders()`**  
  rigenera le triplette a ogni epoca, mantenendo la stessa API del contrastive loader.

- **`_generate_triplets()`**  
  unico punto responsabile della generazione:
  - genera tutte le combinazioni anchor–positive,
  - campiona fino a `samples_per_place`,
  - seleziona negativi da place diversi,
  - supporta permutations o combinations,
  - è agnostica rispetto al dataset e al task.

- **`TriCombinationDataset`**  
  dataset minimale che si occupa solo del caricamento delle tre immagini, sfruttando la logica comune del `BaseDataset`.

Il risultato è un modulo autonomo e intercambiabile.

### Integrazione nel flusso di training

`train.py` ora non contiene più alcuna logica di creazione delle triplette.  
Il ciclo di training:

- chiama `get_dataloaders(...)` all’inizio,  
- chiama `refresh_dataloaders(...)` a ogni epoca,  
- utilizza la stessa interfaccia sia per Triplet che per InfoNCE.

Questo separa in modo netto:

- **logica del training**  
- **logica del sampling**  
- **logica della configurazione**

e permette di sostituire il metodo di apprendimento senza toccare nessun’altra parte della pipeline.

### Possibilità introdotte dal refactor

Grazie alla generalizzazione del generatore di triplette ora è possibile:

- scegliere **qualsiasi numero di triplette per place**, sia nel training che nella validation;  
- usare **dataset arbitrari** (GTA, Alderley, GSV, Tokyo, ecc.) senza adattamenti manuali;  
- sostituire la Triplet Loss con altre tecniche mantenendo lo stesso sistema di sampling;  
- ottenere una pipeline modulare dove ogni componente è isolato e testabile.

### Risultati con l’estensione a 2 triplette per place per training e validation

| k  | Recall |
|----|---------|
| 1  | 0.4126  |
| 5  | 0.6217  |
| **10** | **0.7019**  |
| 50 | 0.8878  |

## 4. Scelta dell’optimizer e normalizzazione immagini

È stato eseguito un confronto diretto tra tre ottimizzatori:

- **Adam**  
- **AdamV**  
- **AdamW**

**AdamW** ha prodotto un miglioramento leggero ma costante risultando il più stabile tra i tre.

Parallelamente è stata introdotta la **normalizzazione delle immagini** (prima non sfruttata efficacemente nel task daynight).  
Nel contesto VPR, al contrario, la normalizzazione fornisce un beneficio misurabile:

| k  | Recall |
|----|---------|
| 1  | 0.4134  |
| 5  | 0.6346  |
| **10** | **0.7179**  |
| 50 | 0.8974  |

Questi esperimenti sono stati volutamente compatti e mirati:  
non è stato introdotto nuovo codice complesso, ma solo micro-ottimizzazioni per identificare l’optimizer più adatto e verificare l’impatto della normalizzazione nel setup VPR.


## 5. Ricerca dei migliori campioni per place
Dopo vari esperimenti, la combinazione preferibile è risultata:

- `train_samples_per_place = 4`  
- `valid_samples_per_place = 2`

risultando in:

| k | Recall |
|---|--------|
| 1 | 0.4358 |
| 5 | 0.6474 |
| **10** | **0.7211** |
| 50 | 0.9038 |

## 6. Generalizzazione del metodo di learning (Triplet → InfoNCE)

Dopo la completa modularizzazione delle triplette, la pipeline è stata estesa per supportare un secondo metodo di apprendimento: **InfoNCE**.  
L’obiettivo era verificare se, nel contesto VPR, un loss contrastivo moderno potesse superare le performance del classico approccio Triplet.

La selezione del metodo è ora immediata, tramite YAML:

- `learning_method: "triplet"`  
- `learning_method: "infonce"`

### Introduzione del modulo contrastive

Il nuovo file `contrastive_loader.py` contiene l’intera pipeline InfoNCE/SupCon:

- **`SupConDataset`**  
  dataset che restituisce batch strutturati come gruppi di immagini dello stesso place.

- **`_generate_supcon_batches()`**  
  genera batch contrastivi mantenendo `samples_per_place` e `places_per_batch` configurabili.

- **`get_contrastive_dataloaders()`** e **`refresh_contrastive_dataloaders()`**  
  analoghi alle versioni triplet, ma ottimizzati per il loss contrastivo.

- **`get_supcon_loss()`**  
  implementazione completa della NT-Xent (InfoNCE), con temperature configurabile.

L’integrazione con il training è stata strutturata in modo da replicare esattamente il comportamento delle triplette: API identiche, refresh a ogni epoca, stessa gestione dataset, stessa compatibilità con gli scheduler.

### Motivazione del metodo

InfoNCE si presta bene a VPR perché:

- opera su batch più ricchi,  
- sfrutta meglio il numero di samples per place,  
- stabilizza il training quando `samples_per_place > 2`,  
- sfrutta la normalizzazione interna e la temperatura per controllare le similarità.

La pipeline era limitata alla Triplet Loss; questa estensione ha richiesto implementazione, debugging del batching, tuning dei parametri e confronto empirico con le configurazioni esistenti.

### Risultati iniziali

**Test base InfoNCE:**

- 4 samples train  
- 2 samples val  
- 32 places per batch  
- AdamW  
- Normalize attivo  
- Temperature 0.05  

| k | Recall |
|---|--------|
| 1 | 0.3717 |
| 5 | 0.5769 |
| **10** | **0.6891** |
| 50 | 0.8942 |

### Test estesi

Ampliando i parametri:

- 8 samples train  
- 3 samples val  
- 48 places per batch  

| k | Recall |
|---|--------|
| 1 | 0.4070 |
| 5 | 0.6538 |
| **10** | **0.7371** |
| 50 | 0.9262 |

### Considerazioni sul miglioramento

L’aumento è significativo e conferma che InfoNCE si adatta meglio alla struttura dei dataset VPR, dove molte viste dello stesso luogo possono essere sfruttate per creare embedding più robusti.  
Molto del lavoro in questo capitolo è consistito nel:

- comprendere e implementare accuratamente InfoNCE/SupCon,  
- rifinire la generazione dei batch contrastivi,  
- testare combinazioni numeriche (samples, places_per_batch, temperature),  
- validare che il tutto fosse compatibile con la pipeline già generalizzata.

L’infrastruttura ora supporta due metodi di apprendimento completamente intercambiabili, permettendo ulteriori esplorazioni future senza modifiche strutturali alla pipeline.

## 7. Hard Negative Mining — Implementazione, Accelerazioni e Risultati

L’obiettivo era introdurre una forma di hard negative mining nella Triplet Loss, selezionando per ogni anchor il negativo più vicino nello spazio degli embedding. La procedura richiedeva il calcolo degli embedding globali e la ricerca del negativo più simile tramite indice ANN.

### Implementazione

1. **Computazione globale degli embedding**  
   Implementata in `embedding.py` (`compute_global_embeddings()`), con:
   - batch su GPU  
   - mixed precision  
   - normalizzazione L2  
   - caching completo  

2. **Indice ANN**  
   Gestito tramite `NearestNeighbors` (cosine) per individuare i candidati più vicini.

3. **Caching**  
   Sviluppato in `cache_manager.py`, con refresh controllato da `hard_negative_cache_refresh_rate`.

4. **Accelerazioni**  
   - semihard negatives (top-20)  
   - fallback lineare su top-k limitato  
   - pre-filtraggio per place  

5. **Integrazione nel dataloader**  
   In `triplet_loader.py`, `_generate_triplets()` seleziona negativi randomici o hard a seconda delle impostazioni.

### Costi computazionali

Nonostante caching, ANN, semihard negatives e batch ottimizzati, i tempi di training aumentano sensibilmente rispetto alla selezione randomica dei negativi.

### Risultati

| k | Recall |
|---|--------|
| 1 | 0.4290 |
| 5 | 0.6512 |
| **10** | **0.6945** |
| 50 | 0.8876 |

I valori sono molto simili alla selezione randomica, senza miglioramenti misurabili.

### Conclusione

Dato il costo elevato e l’assenza di benefici, la tecnica è stata accantonata. Potrà essere reintrodotta solo in fasi successive della pipeline o con loss più sensibili agli hard negatives.


## Risultati principali

| Configurazione | Metodo | Recall@1 | Recall@5 | **Recall@10** | Recall@50 |
|----------------|--------|------------|------------|------------|------------|
| 1 train / 1 val (**baseline VPR**) | Triplet | 0.4011 | 0.5987 | 0.6402 | 0.8147 |
| 2 train / 2 val | Triplet | 0.4126 | 0.6217 | 0.7019 | 0.8878 |
| Normalizzazione | Triplet | 0.4134 | 0.6346 | 0.7179 | 0.8974 |
| 4 train / 2 val | Triplet | 0.4358 | 0.6474 | 0.7211 | 0.9038 |
| Hard Negative   | Triplet | 0.4290 | 0.6512 | 0.6945 | 0.8876 |
| 4 train / 2 val | InfoNCE | 0.3717 | 0.5769 | 0.6891 | 0.8942 |
| 8 train / 3 val | InfoNCE | 0.4070 | 0.6538 | 0.7371 | 0.9262 |

---

## Analisi dei risultati

1. Early stopping riduce enormemente i tempi di training con qualità invariata o migliore.  
2. La pipeline VPR generalizzata permette esperimenti più realistici rispetto al daynight.  
3. Le triplette multiple hanno dato un boost immediato alla performance.  
4. AdamW risulta migliore di Adam in modo costante.  
5. La normalizzazione è decisiva nel contesto VPR.    
6. L’aumento dei samples per place migliora fino a un plateau.  
7. La pipeline è ora più modulare, estendibile e pronta a tecniche più avanzate.

---

## Combinazioni più performanti

| Metodo | Parametri principali | Recall@1 | Recall@5 | **Recall@10** | Recall@50 |
|--------|-----------------------|------------|---|---|---|
|**Triplet**| 4 train, 2 val | 0.4358 | 0.6474 | **0.7211** | 0.9038 |
| **InfoNCE** | 8 train, 3 val, 48 places per batch | 0.4070 | 0.6538 | **0.7371** | 0.9262 |

---

## Conclusioni

- Il training è passato **da 400 a ~25 epoche** mantenendo o migliorando la performance.  
- La **pipeline** è ora completamente **generalizzata** e configurabile via YAML.  
- Triplette multiple, **AdamW** e **normalizzazione** hanno **migliorato** in modo incrementale la qualità.  
- InfoNCE rappresenta un'ottima alternativa da esplorare in termini di performance e flessibilità; la **coesistenza** delle loss **Triplet e InfoNCE** sarà sfruttata per valutarne l'efficacia in diverse condizioni.
- La pipeline è robusta, modulare e pronta a estensioni avanzate.

---

## Sviluppi futuri proposti per l’Esperimento 3

- Continuare con ResNet50, concentrandosi sul **preprocessing delle immagini**.  
- Introduzione di tecniche **gta2real** per ridurre il domain gap.  
- Adozione di augmentation più avanzate per migliorare la robustezza del modello.
