# Esperimento 003 — Preprocessing Avanzato e Riduzione del Domain Gap

## Descrizione generale

L'attenzione dell'**Esperimento 003** si sposta dall'architettura al **dato**.

L'ipotesi alla base di questa fase è che il collo di bottiglia attuale non risieda più nella capacità di apprendimento del modello, ma nella qualità e nella rappresentatività degli input. Questo esperimento stabilisce la "baseline infrastrutturale" necessaria per i futuri test di Data Augmentation e Domain Adaptation.

---

## 1. Refactor e Modularizzazione del Preprocessing

Nelle iterazioni precedenti, la logica di trasformazione delle immagini (crop, resize, normalizzazione) era "hardcoded" all'interno delle classi `Dataset` (es. in `__getitem__`). Questo approccio rendeva complesso testare rapidamente nuove combinazioni di preprocessing senza modificare il core del codice.

È stato quindi effettuato un **refactor completo**, estraendo tutta la logica di manipolazione in un modulo dedicato ed esterno.

### Architettura "Mattoncini e Orchestratore"

La nuova struttura si basa su una netta separazione di responsabilità, implementata tramite un set di funzioni private ("mattoncini") e una funzione pubblica di gestione ("orchestratore").

**Componenti base (Funzioni private):**
Sono state isolate le operazioni atomiche, rendendole stateless e puramente funzionali:
* `_load_image_as_tensor`: Caricamento e conversione immediata in tensore.
* `_get_random_crop_coords`: Calcolo delle coordinate, separato dall'applicazione del taglio.
* `_apply_crop` / `_center_crop`: Applicazione fisica del ritaglio.
* `_resize` / `_normalize`: Adattamento dimensionale e statistico.

**L'Orchestratore (`preprocess_data`):**
È stato introdotto un unico entry-point `preprocess_data(path, params, previous_crop=None)` che agisce come controller:
1.  Legge la configurazione direttamente dai parametri **YAML**.
2.  Gestisce la logica condizionale (es. scelta tra *random crop* e *center crop*).
3.  **Gestisce la coerenza spaziale**: accetta un parametro opzionale `previous_crop`, fondamentale per le architetture Siamesi/Triplet, garantendo che, se necessario, la stessa porzione casuale venga ritagliata su immagini diverse (o sulla stessa immagine in epoche diverse).

### Configurazione via YAML

Grazie a questo refactor, il preprocessing è ora interamente pilotabile dal file di configurazione, senza interventi sul codice Python. Esempio di struttura supportata:

```yaml
preprocessing:
  use_random_crop: true
  use_center_crop: false
  target_height: 224
  target_width: 224
  normalize: true
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```
Questa modifica abilita **Test A/B** immediati e garantisce che l'aggiunta di filtri complessi (come quelli analizzati nella sezione successiva) non alteri la logica di caricamento dati.

---

## 2. Analisi Preliminare Data Augmentation: Random Grayscale

Una volta consolidata l'infrastruttura di preprocessing, è stato condotto un primo esperimento mirato alla riduzione del *Domain Gap* tra il dataset sintetico (GTA V) e quello reale (GSV-Cities).

### Ipotesi
L'ipotesi di partenza è che la rete neurale possa sovra-adattarsi ("overfitting") alle caratteristiche cromatiche specifiche del motore grafico di GTA V (palette colori satura, illuminazione perfetta). Introducendo una conversione casuale in scala di grigi (`Random Grayscale`), si intende forzare il modello a concentrarsi sulle caratteristiche geometriche e strutturali della scena (forme degli edifici, skyline, layout stradale), teoricamente più robuste al cambio di dominio.

### Configurazione Esperimento
* **Tecnica:** Random Grayscale (mantenendo 3 canali in output per compatibilità ResNet).
* **Probabilità di applicazione ($p$):** 0.2 (20% delle immagini di training).
* **Dataset Training:** GTA V.
* **Dataset Validation:** GSV-Cities (Valid).

### Risultati e Confronto
Di seguito il confronto delle metriche di Recall@K tra la Baseline (nessuna augmentation colore) e l'esperimento con Grayscale.

| Metrica | Baseline (%) | Grayscale ($p=0.2$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 45.83% | -0.64% |
| **Top-5** | **67.31%** | 66.02% | -1.29% |
| **Top-10** | **75.96%** | 75.64% | -0.32% |
| **Top-50** | **93.59%** | 91.67% | -1.92% |

### Conclusioni Parziali
L'esperimento ha evidenziato una **lieve regressione** delle performance.
* Il calo contenuto sulla **Top-1 (-0.64%)** indica che le feature geometriche sono state apprese, ma la perdita dell'informazione cromatica ha ridotto la capacità discriminativa fine.
* Il calo più marcato sulla **Top-50 (-1.92%)** suggerisce che il colore agisce come importante fattore di disambiguazione per i candidati "difficili".

Alla luce di questi dati, la tecnica viene momentaneamente **disabilitata** per procedere con test su trasformazioni che agiscano sulla texture (es. *Gaussian Blur*) piuttosto che sulla cromia, mantenendo il codice integrato nell'orchestratore per futuri utilizzi combinati.

---

## 3. Analisi Data Augmentation: Gaussian Blur

Dopo i risultati non conclusivi del *Grayscale*, si è tentato di ridurre il domain gap agendo sulla "texture" delle immagini. Le immagini sintetiche (GTA) sono caratterizzate da una nitidezza innaturale e bordi estremamente definiti rispetto alle immagini reali (GSV), spesso affette da motion blur o focus imperfetto.

### Configurazione Esperimento
* **Tecnica:** Gaussian Blur (Sfocatura Gaussiana).
* **Probabilità ($p$):** 0.5 (50% delle immagini).
* **Kernel Size:** [5, 5].
* **Sigma:** [0.1, 2.0].

### Risultati
L'esperimento ha portato a un **degrado significativo** delle performance rispetto alla Baseline.

| Metrica | Baseline (%) | Gaussian Blur ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 43.59% | **-2.88%** |
| **Top-5** | **67.31%** | 66.03% | -1.28% |
| **Top-10** | **75.96%** | 75.64% | -0.32% |
| **Top-50** | **93.59%** | 91.99% | -1.60% |

### Conclusioni
L'ipotesi che la sfocatura aiutasse la generalizzazione è stata smentita dai dati.
La rimozione dei dettagli ad alta frequenza ha impedito alla rete di apprendere feature geometriche fini (es. pattern architettonici, contorni netti degli edifici) necessarie per la discriminazione fine dei luoghi. Il costo computazionale aggiuntivo (raddoppio del tempo per epoca) non è giustificato.
La tecnica viene **disabilitata**.

---

## 4. Analisi Data Augmentation: Random Horizontal Flip

Dopo aver scartato le modifiche cromatiche (Grayscale) e di texture (Blur), si è testata una trasformazione geometrica "lossless": il ribaltamento orizzontale (*Random Horizontal Flip*).

### Ipotesi
L'obiettivo era rendere il modello invariante rispetto alla direzione laterale, aumentando artificialmente la diversità del training set senza degradare la qualità dell'immagine.

### Risultati
Anche in questo caso si registra una lieve regressione rispetto alla Baseline.

| Metrica | Baseline (%) | Horizontal Flip ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 46.15% | -0.32% |
| **Top-5** | **67.31%** | 66.02% | -1.29% |
| **Top-10** | **75.96%** | 75.32% | -0.64% |
| **Top-50** | **93.59%** | 93.27% | -0.32% |

### Conclusioni
Il leggero calo indica che nella *Visual Place Recognition* urbana la geometria laterale è una feature discriminante (es. posizione della carreggiata, disposizione degli edifici rispetto alla strada). L'augmentation introduce un "rumore geometrico" che non aiuta il trasferimento dal dominio sintetico a quello reale.
La tecnica viene **disabilitata**.

---

## 5. Analisi Data Augmentation: Random Resized Crop

Come ulteriore tentativo geometrico, si è introdotto il *Random Resized Crop* (RRC), che simula variazioni di scala (zoom) e di rapporto d'aspetto.

### Ipotesi
L'idea era rendere la rete robusta a variazioni di distanza e scala, insegnando che un luogo rimane identico anche se visto più da vicino (zoom-in) o con deformazioni prospettiche lievi.

### Risultati
L'esperimento ha causato un peggioramento delle performance su tutte le metriche.

| Metrica | Baseline (%) | Resized Crop (Scale 0.4-1.0) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 44.55% | -1.92% |
| **Top-5** | **67.31%** | 64.74% | -2.57% |
| **Top-10** | **75.96%** | 74.36% | -1.60% |
| **Top-50** | **93.59%** | 92.95% | -0.64% |

### Conclusioni
La perdita di contesto globale causata dal crop ha impattato negativamente. Nella *Place Recognition*, la disposizione relativa degli elementi (es. "l'albero è a sinistra del palazzo") è cruciale. RRC altera o rimuove queste relazioni spaziali.
La tecnica viene **disabilitata**.

---

## 6. Analisi Data Augmentation: Random Erasing

Per simulare le occlusioni tipiche degli ambienti urbani reali (pedoni, auto, vegetazione), si è testato il *Random Erasing*, che sovrascrive rettangoli casuali dell'immagine con valori nulli (nero).

### Ipotesi
L'obiettivo era forzare la rete a riconoscere i luoghi basandosi su porzioni parziali della scena, aumentandone la robustezza contro gli ostacoli visivi presenti nel dataset di test (GSV).

### Risultati
L'introduzione di occlusioni artificiali ha degradato le performance, indicando che la perdita di informazione visiva ostacola l'apprendimento delle feature discriminative.

| Metrica | Baseline (%) | Random Erasing ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 44.23% | -2.24% |
| **Top-5** | **67.31%** | 65.38% | -1.93% |
| **Top-10** | **75.96%** | 74.68% | -1.28% |
| **Top-50** | **93.59%** | 91.67% | -1.92% |

### Conclusioni
La tecnica ha rimosso dettagli utili senza offrire benefici di generalizzazione. Probabilmente, dato il domain gap già ampio, la rete necessita di visuali pulite per ancorare le feature sintetiche a quelle reali.
La tecnica viene **disabilitata**.

---

## 7. Analisi Data Augmentation: Color Jitter (Photometric Distortions)

Come ultimo tentativo di augmentation, si è testata la manipolazione fotometrica tramite *Color Jitter* (luminosità, contrasto, saturazione), mantenendo parametri conservativi per evitare distorsioni irrealistiche.

### Configurazione Esperimento
* **Brightness, Contrast, Saturation:** 0.3 ($\pm$30%).
* **Hue:** 0.05 (variazione minima).
* **Impatto Computazionale:** Rallentamento significativo del training (CPU bottleneck).

### Risultati
Anche l'alterazione dei colori ha portato a un calo delle performance, seppur contenuto.

| Metrica | Baseline (%) | Color Jitter (Soft) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 45.51% | -0.96% |
| **Top-5** | **67.31%** | 65.71% | -1.60% |
| **Top-10** | **75.96%** | 75.32% | -0.64% |
| **Top-50** | **93.59%** | 92.95% | -0.64% |

### Conclusioni Generali Fase Preprocessing
Al termine di 6 esperimenti mirati (Grayscale, Blur, Flip, Resized Crop, Erasing, Jitter), i dati indicano chiaramente che **le tecniche standard di Data Augmentation non sono efficaci per il task Syn2Real (GTA $\to$ GSV)** con l'attuale architettura.
Ogni alterazione dell'input sintetico ha ridotto la capacità discriminativa del modello. La pulizia del dato GTA sembra essere un asset fondamentale per l'apprendimento delle feature strutturali, più che un difetto da mascherare.
La pipeline di produzione verrà quindi configurata sulla **Baseline (Minimal Preprocessing)**.