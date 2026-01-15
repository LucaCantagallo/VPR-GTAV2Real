# Esperimento 004 — Model Selection e Scaling dell'Architettura

## Obiettivo dell'Esperimento
Dopo aver appurato nell'Esperimento 003 che il preprocessing non è sufficiente a colmare il domain gap, l'**Esperimento 004** si pone l'obiettivo di identificare l'architettura (Backbone) ottimale per questo task.
L'ipotesi è che la ResNet50 (Baseline) abbia raggiunto la saturazione della capacità semantica. In questo esperimento verranno testati diversi modelli candidati a parità di condizioni di training.

---

## 1. Aggiornamento Infrastrutturale: Model Factory

Prima di procedere con i test dei candidati, è stato necessario un refactoring del codice per supportare il cambio dinamico della rete ("Model Swapping").

* **Disaccoppiamento:** Creazione della folder `architectures/` per definire i modelli in isolamento.
* **Gestione Dinamica Output:** Il sistema ora si adatta automaticamente alla dimensione delle feature del backbone (es. 2048, 768, etc.) senza modifiche manuali alla testa MLP.
* **Pipeline Parametrica:** Selezione del modello direttamente da `pipeline.yaml`.

---

## 2. Candidate #1: ResNeXt-50 (32x4d)

Il primo candidato testato è stato **ResNeXt-50**, scelto per la sua stretta parentela con la baseline ma con l'aggiunta della "Cardinality" (Grouped Convolutions), teoricamente capace di estrarre feature più diversificate.

### Configurazione Training
* **Modello:** `resnext50_32x4d` (Pesi ImageNet-1k V2).

### Risultati Candidate #1

| Metrica | Baseline (ResNet50) | Candidate #1 (ResNeXt-50) | Delta |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 42.31% | **-4.16%** |
| **Recall@5** | **67.31%** | 60.26% | -7.05% |
| **Recall@10** | **75.96%** | 69.87% | -6.09% |
| **Recall@50** | **93.59%** | 89.42% | -4.17% |

### Analisi e Conclusioni (Candidate #1)
Il candidato **ResNeXt-50 è stato scartato**.
Nonostante la maggiore capacità teorica, il modello ha mostrato una regressione significativa su tutte le metriche.
**Ipotesi del fallimento:**
1.  **Overfitting su pattern sintetici:** La maggiore cardinalità potrebbe aver catturato dettagli ad alta frequenza specifici del render (GTA) che non esistono nel reale (GSV), riducendo la generalizzazione.
2.  **Instabilità BN:** L'architettura ResNeXt soffre maggiormente l'assenza di batch statistici stabili (Batch Size = 1) rispetto alla ResNet classica.

---

## 3. Candidate #2: ResNeSt-50 (Split-Attention)

Il secondo candidato è **ResNeSt-50**.
Dopo il fallimento dell'approccio basato sulla cardinalità (ResNeXt), si è tentato l'approccio basato sull'**Attenzione**. ResNeSt introduce blocchi *Split-Attention* che permettono alla rete di focalizzarsi sulle feature più salienti (es. strutture geometriche) ignorando il rumore di fondo, ideale teoricamente per colmare il domain gap sintetico-reale.

### Configurazione Training
* **Modello:** `resnest50` (Pesi ImageNet-1k).
* **Environment:** Training in Mixed Precision (AMP) per vincoli di memoria hardware.

### Risultati Candidate #2

| Metrica | Baseline (ResNet50) | Candidate #2 (ResNeSt-50) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 41.99% | **-4.48%** |
| **Recall@5** | **67.31%** | 61.22% | -6.09% |
| **Recall@10** | **75.96%** | 70.51% | -5.45% |
| **Recall@50** | **93.59%** | 89.10% | -4.49% |

### Analisi e Conclusioni (Candidate #2)
Anche il candidato **ResNeSt-50 è stato scartato**. I risultati sono addirittura leggermente inferiori a ResNeXt.

---

## 4. Analisi Critica dei Fallimenti e Refactoring Architetturale

Dopo il fallimento dei primi due candidati utilizzati "out-of-the-box" (ovvero mantenendo l'Average Pooling nativo), è stata condotta un'analisi più profonda.
Il problema identificato non risiede nella capacità estrattiva delle backbone, ma nel **collo di bottiglia dell'Aggregatore**.

* **Il problema dell'Average Pooling:** Le implementazioni standard di ResNet/ResNeSt terminano con un *Global Average Pooling* che media tutte le feature spaziali. In un task di VPR Sim-to-Real, questo "lava via" i dettagli discriminativi geometrici (es. spigoli unici), mischiandoli con feature di texture (cielo, strada sintetica) che variano troppo tra i domini.
* **La Soluzione (GeM Pooling):** Si è deciso di implementare il **Generalized Mean Pooling (GeM)**. Il GeM (con parametro $p$ apprendibile) agisce come una via di mezzo tra MaxPool e AvgPool, permettendo alla rete di focalizzarsi sulle attivazioni spaziali più intense (i landmark) e ignorando il rumore di fondo.

### Interventi sul Codice
Per abilitare il GeM, è stato necessario un ulteriore refactoring strutturale:
1.  **Chirurgia delle Backbone:** Le funzioni di build (`build_resnest50`, etc.) sono state modificate per "tagliare" la rete **prima** dell'`AvgPool`.
    * *Output precedente:* Vettore 1D ($2048$).
    * *Nuovo Output:* Tensore Spaziale 3D ($2048 \times 7 \times 7$).
2.  **Modulo Aggregator:** Implementazione modulare di `GeM` in `aggregators.py`, inserita nella pipeline subito dopo la backbone.

---

## 5. Candidate #3: ResNeSt-50 + GeM Pooling

Con la nuova architettura modulare, si è lanciato il training del terzo candidato: **ResNeSt-50 accoppiata con GeM Pooling**.

Per permettere il training di questa architettura, è stato necessario ridurre drasticamente il parallelismo nel file di configurazione (`pipeline.yaml`), accettando gradienti potenzialmente più rumorosi pur di sbloccare l'addestramento:

| Parametro | Valore Precedente | Valore Attuale (Fix) |
| :--- | :---: | :---: |
| **Train Samples per Place** | 8 | **4** |
| **Contrastive Places per Batch** | 48 | **5** |
| **Immagini Totali per Batch** | 384 | **20** |

### Risultati Candidate #3 (ResNeSt-50 + GeM)

| Metrica | Baseline (ResNet50) | Candidate #3 (ResNeSt+GeM) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 31.41% | **-15.06%** |
| **Recall@5** | **67.31%** | 53.21% | -14.10% |
| **Recall@10** | **75.96%** | 61.54% | -14.42% |
| **Recall@50** | **93.59%** | 83.33% | -10.26% |

### Analisi e Conclusioni (Candidate #3)
L'esperimento con **ResNeSt-50 + GeM** ha registrato un **crollo verticale delle performance**, risultando la configurazione peggiore testata finora.

---

## 4. Candidate #3: ConvNeXt-Tiny

Abbandonando la famiglia ResNet, si è testata **ConvNeXt-Tiny**. Questa architettura rappresenta lo stato dell'arte delle CNN moderne (2022), progettata per competere con i Vision Transformers integrando modernizzazioni strutturali (kernel 7x7, LayerNorm, GELU, patchify stem).

### Configurazione Training
* **Modello:** `convnext_tiny` (Pesi ImageNet-1k V1).
* **Batch Size:** 384 immagini (Configurazione Full: `48 places x 8 samples`).
* **Output Dim:** 768 (Dimensione nativa, senza compressione o espansione artificiale).
* **Aggregatore:** Average Pooling (Standard).

### Risultati Candidate #3

| Metrica | Baseline (ResNet50) | Candidate #3 (ConvNeXt-Tiny) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 43.59% | **-2.88%** |
| **Recall@5** | **67.31%** | 63.46% | -3.85% |
| **Recall@10** | **75.96%** | 74.36% | -1.60% |
| **Recall@50** | **93.59%** | 89.42% | -4.17% |

### Analisi e Conclusioni (Candidate #3)
Il candidato **ConvNeXt-Tiny** ha ottenuto prestazioni **lievemente inferiori alla baseline** (-2.88% a R@1).
Il risultato è controintuitivo considerando la superiorità architetturale di ConvNeXt su ImageNet, ma evidenzia tre criticità specifiche del task VPR *Synth-to-Real*:

1.  **Collo di Bottiglia Dimensionale:** La Baseline proietta le feature in uno spazio a **2048** dimensioni. ConvNeXt-Tiny lavora nativamente a **768**. Questa compressione intrinseca ($2.6\times$ inferiore), combinata con l'Average Pooling, potrebbe causare una perdita di discriminabilità sui dettagli fini necessari per distinguere luoghi simili.
2.  **Il Limite dell'Average Pooling:** ConvNeXt estrae feature ad alto livello semantico (oggetti interi). L'uso dell'*Average Pooling* (standard in classificazione) media queste feature con il "rumore semantico" del contesto (es. cielo e asfalto sintetici di GTA), diluendo l'informazione utile per la geolocalizzazione.
3.  **Drop-in Replacement:** L'utilizzo degli stessi iperparametri della ResNet (LR, Weight Decay) su un'architettura che usa LayerNorm e dinamiche di gradiente diverse potrebbe non essere ottimale senza un tuning specifico.

---

## 5. Candidate #4: ConvNeXt-Tiny + GeM Pooling

Dopo il risultato quasi incoraggiante del Candidate #3 (ConvNeXt + AvgPool, che perdeva solo il 3% circa), si è tentato di risolvere il problema del "rumore di sfondo" sostituendo l'Average Pooling con il **GeM Pooling**.
L'ipotesi era che il GeM avrebbe aiutato la ConvNeXt a focalizzarsi sui landmark strutturali, ignorando gli artefatti sintetici.

### Configurazione Training
* **Modello:** `convnext_tiny` (Pesi ImageNet-1k V1).
* **Aggregatore:** **GeM (Generalized Mean Pooling)**, con $p$ inizializzato a 3.0.
* **Batch Size:** 384 immagini (Configurazione Full).
* **Output Dim:** 768.

### Risultati Candidate #4

| Metrica | Baseline (ResNet50) | Candidate #4 (ConvNeXt+GeM) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 22.76% | **-23.71%** |
| **Recall@5** | **67.31%** | 44.87% | -22.44% |
| **Recall@10** | **75.96%** | 51.60% | -24.36% |
| **Recall@50** | **93.59%** | 75.32% | -18.27% |

### Analisi e Conclusioni (Candidate #4)
L'esperimento è risultato in un **fallimento critico**. Le performance sono crollate a livelli inaccettabili (quasi dimezzata la Recall@1 rispetto alla baseline).

**Diagnosi del Crollo:**
Questo risultato evidenzia una **incompatibilità strutturale** tra le feature di ConvNeXt e l'aggregazione GeM:
1.  **Natura delle Feature:** A differenza delle ResNet (che producono attivazioni "spiky", ideali per GeM/Max pooling), ConvNeXt utilizza LayerNorm e attivazioni GELU che tendono a produrre mappe di feature più "lisce" e distribuite spazialmente.
2.  **Soppressione dell'Informazione:** Applicando un pooling che enfatizza i picchi (GeM con $p=3$), la rete ha probabilmente soppresso informazioni contestuali critiche che erano distribuite in modo uniforme sulla feature map, distruggendo la rappresentazione semantica del luogo.

---

## 6. Candidate #5: DINOv2 (ViT-B/14 + GeM) — La Svolta

Dopo i fallimenti delle architetture CNN moderne (ConvNeXt) e delle varianti ResNet avanzate, si è optato per un cambio di paradigma radicale:
1.  **Architettura:** Passaggio dalle CNN ai **Vision Transformers (ViT)**.
2.  **Learning Paradigm:** Passaggio dal Supervised Learning (ImageNet labels) al **Self-Supervised Learning (SSL)**.

Il candidato scelto è **DINOv2** (versione Base, ViT-B/14). L'ipotesi è che, essendo addestrato senza etichette per massimizzare la corrispondenza tra view diverse, DINOv2 apprenda rappresentazioni robuste basate sulla **geometria e semantica strutturale** piuttosto che sulle texture fini. Questo è cruciale per il task *GTA-to-Real*, dove la geometria è consistente ma le texture differiscono (Domain Gap).

### Implementazione Tecnica (Token Strategy)
Per adattare un ViT al task di VPR (che beneficia del GeM Pooling), non si è utilizzato l'output standard (il *CLS token*, che è un vettore 1D).
È stata implementata una strategia ibrida:
* **Patch Extraction:** Si estraggono i *Patch Tokens* ($N$ vettori).
* **Spatial Reshape:** I token vengono rimodellati in un tensore 3D pseudo-spaziale ($B \times 768 \times 16 \times 16$), simulando una feature map.
* **Aggregazione:** Su questa mappa viene applicato il **GeM Pooling**.

### Configurazione Training
* **Modello:** `dinov2_vitb14` (Pesi ufficiali Facebook Research).
* **Backbone Status:** **Frozen** (`trainable_from_layer: null`). Si addestra solo la testa (Linear Probing).
* **Aggregatore:** GeM Pooling.
* **Output Dim:** 512 (Projector MLP da 768 a 512).

### Risultati Candidate #5 (DINOv2)

| Metrica | Baseline (ResNet50) | Candidate #5 (DINOv2) | Delta vs Baseline |
| :--- | :---: | :---: | :---: |
| **Recall@1** | 46.47% | **81.09%** | **+34.62%** |
| **Recall@5** | 67.31% | **90.06%** | **+22.75%** |
| **Recall@10** | 75.96% | **91.99%** | **+16.03%** |
| **Recall@50** | 93.59% | **96.47%** | +2.88% |

### Analisi e Conclusioni (Candidate #5)
Il risultato è **eccezionale**. DINOv2 non solo batte la baseline, ma la distrugge, portando la Recall@1 da un mediocre 46% a un eccellente **81%**.

**Perché ha funzionato dove gli altri hanno fallito?**
1.  **Robustezza al Domain Shift:** Mentre ResNet e ConvNeXt (Supervised) cercavano pattern di texture (asfalto realistico vs asfalto finto), DINOv2 (SSL) "comprende" la scena 3D. Riconosce che un palazzo è un palazzo indipendentemente dal rendering grafico.
2.  **Efficacia del Frozen Backbone:** Il fatto che la backbone fosse congelata ha impedito al modello di *overfittare* sullo stile grafico di GTA V. Abbiamo sfruttato la potenza delle feature pre-addestrate pure, adattando solo la proiezione geometrica (GeM + Head).
3.  **Spatial Tokens vs CLS:** L'intuizione di usare i patch token spaziali con GeM (invece del CLS token standard dei ViT) ha permesso di mantenere la discriminabilità spaziale necessaria per il Place Recognition.

----

## 7. Ottimizzazione di DINOv2: Scaling e Resolution Boost

Dopo aver identificato **DINOv2 (ViT-B/14)** come l'architettura vincente (Recall@1 > 80%), l'esperimento si è esteso per indagare se l'aumento della capacità computazionale (Scaling) o della densità informativa (Resolution) potesse spingere ulteriormente le performance verso il 90% in Top-1.

Sono state confrontate 4 configurazioni della famiglia DINOv2, mantenendo fisse le condizioni di training (Backbone Frozen, GeM Pooling, Dataset GTA):

1.  **Baseline (Exp 1):** DINOv2 Base (ViT-B/14) @ 224x224.
2.  **Resolution Boost (Exp 2):** DINOv2 Base (ViT-B/14) @ **322x322** (Aumento dei patch token da 256 a 529).
3.  **Model Scaling (Exp 3):** DINOv2 **Large** (ViT-L/14) @ 224x224.
4.  **Max Config (Exp 4):** DINOv2 Large (ViT-L/14) @ 322x322.

### Tabella Comparativa Risultati

Di seguito il confronto diretto tra la vecchia baseline (ResNet50) e le varianti DINOv2. Le metriche di interesse primario (Top-5 e Top-10) sono evidenziate.

| Configurazione | Backbone | Risoluzione | Recall@1 | Recall@5 | Recall@10 | Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Old Baseline** | ResNet50 | 224 | 46.47% | 67.31% | 75.96% | Saturazione performance |
| **DINO Base** | ViT-B/14 | 224 | **81.09%** | 90.06% | 91.99% | **Best Trade-off** |
| **DINO Res+** | ViT-B/14 | 322 | 80.77% | **91.35%** | 92.63% | Best R@5/10 (ex-aequo) |
| **DINO Large** | ViT-L/14 | 224 | 79.81% | 91.03% | **93.27%** | Lieve calo in R@1 |
| **DINO Max** | ViT-L/14 | 322 | 79.17% | 89.74% | 92.95% | Regressione |

### Analisi dei Trend

#### 1. Impatto della Risoluzione (224 vs 322)
Passando da 224px a 322px con il modello Base (Exp 1 vs Exp 2), osserviamo un comportamento ibrido:
* **Top-1 (Recall esatta):** Si nota una lievissima flessione (-0.32%).
* **Top-5 / Top-10 (Retrieval robusto):** Si registra un miglioramento costante (+1.29% sulla R@5).
**Interpretazione:** Aumentare la risoluzione incrementa il numero di token su cui il GeM può lavorare ($16 \times 16 \to 23 \times 23$). Questo aiuta la rete a recuperare candidati corretti nella "shortlist" (Top-5/10) grazie a dettagli più fini, ma introduce potenzialmente un leggero "rumore" geometrico che disturba il match perfetto (Top-1) quando il domain gap è elevato.

#### 2. Impatto dello Scaling (Base vs Large)
L'utilizzo del modello **Large** (Exp 3 e 4) ha portato a risultati controintuitivi, segnando una regressione sulla Recall@1 (~79%) rispetto alla versione Base (~81%).
**Diagnosi:**
* **Overfitting Semantico:** Il modello Large possiede una capacità di astrazione semantica superiore (addestrato su dataset più vasti). Tuttavia, mantenendolo **congelato** (Frozen), le sue feature potrebbero essere *troppo* specializzate su concetti di alto livello (oggetti complessi) e meno sensibili alla geometria pura degli edifici di GTA necessaria per questo task specifico.
* **Curse of Dimensionality:** Proiettare feature più complesse (dimensione 1024 del Large) nello stesso spazio latente ridotto (512) tramite MLP lineare potrebbe aver causato una perdita di informazione maggiore rispetto alla proiezione dal modello Base (768).

### Conclusioni Definitive e Selezione Modello
L'analisi dimostra che nel contesto *Synthetic-to-Real* con risorse vincolate:
1.  **Bigger is not always Better:** Aumentare la dimensione del modello senza Fine-Tuning non porta benefici diretti.
2.  **Il Vincitore:** La configurazione **DINOv2 Base (ViT-B/14)** risulta la più robusta.
    * La versione a **224px** è preferibile per efficienza computazionale e massima precisione R@1.
    * La versione a **322px** è una valida alternativa se si vuole massimizzare la R@5 a costo di maggiore VRAM.

Per il prosieguo del progetto, si conferma l'utilizzo di **DINOv2 Base**.