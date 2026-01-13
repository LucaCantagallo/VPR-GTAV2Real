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
Questa modifica abilita **Test A/B** immediati e garantisce che l'aggiunta di filtri complessi non alteri la logica di caricamento dati.

---

## 2. Analisi Preliminare Data Augmentation: Random Grayscale

Una volta consolidata l'infrastruttura di preprocessing, è stato condotto un primo esperimento mirato alla riduzione del *Domain Gap* tra il dataset sintetico (GTA V) e quello reale (GSV-Cities).

### Ipotesi
L'ipotesi di partenza è che la rete neurale possa sovra-adattarsi ("overfitting") alle caratteristiche cromatiche specifiche del motore grafico di GTA V (palette colori satura, illuminazione perfetta). Introducendo una conversione casuale in scala di grigi (`Random Grayscale`), si intende forzare il modello a concentrarsi sulle caratteristiche geometriche e strutturali della scena (forme degli edifici, skyline, layout stradale), teoricamente più robuste al cambio di dominio.

### Configurazione Esperimento
* **Tecnica:** Random Grayscale (mantenendo 3 canali in output per compatibilità ResNet).
* **Probabilità di applicazione ($p$):** 0.2 (20% delle immagini di training).

### Risultati
Di seguito il confronto delle metriche di Recall@K tra la Baseline (nessuna augmentation colore) e l'esperimento con Grayscale.

| Metrica | Baseline (%) | Grayscale ($p=0.2$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 45.83% | -0.64% |
| **Top-5** | **67.31%** | 66.02% | -1.29% |
| **Top-10** | **75.96%** | 75.64% | -0.32% |
| **Top-50** | **93.59%** | 91.67% | -1.92% |

**Conclusioni:** Lieve regressione. Il colore agisce come fattore di disambiguazione importante per i candidati difficili. Tecnica disabilitata.

---

## 3. Analisi Data Augmentation: Gaussian Blur

Dopo i risultati non conclusivi del *Grayscale*, si è tentato di ridurre il domain gap agendo sulla "texture" delle immagini. Le immagini sintetiche sono caratterizzate da una nitidezza innaturale e bordi estremamente definiti rispetto alle immagini reali (GSV), spesso affette da motion blur o focus imperfetto.

### Configurazione Esperimento
* **Tecnica:** Gaussian Blur (Sfocatura Gaussiana).
* **Probabilità ($p$):** 0.5 (50% delle immagini).
* **Kernel Size:** [5, 5].
* **Sigma:** [0.1, 2.0].

### Risultati

| Metrica | Baseline (%) | Gaussian Blur ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 43.59% | **-2.88%** |
| **Top-5** | **67.31%** | 66.03% | -1.28% |
| **Top-10** | **75.96%** | 75.64% | -0.32% |
| **Top-50** | **93.59%** | 91.99% | -1.60% |

**Conclusioni:** Degrado significativo. La rimozione dei dettagli ad alta frequenza impedisce l'apprendimento di feature geometriche fini. Tecnica disabilitata.

---

## 4. Analisi Data Augmentation: Random Horizontal Flip

Dopo aver scartato le modifiche cromatiche (Grayscale) e di texture (Blur), si è testata una trasformazione geometrica "lossless": il ribaltamento orizzontale (*Random Horizontal Flip*).

### Ipotesi
L'obiettivo era rendere il modello invariante rispetto alla direzione laterale, aumentando artificialmente la diversità del training set senza degradare la qualità dell'immagine.

### Risultati

| Metrica | Baseline (%) | Horizontal Flip ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 46.15% | -0.32% |
| **Top-5** | **67.31%** | 66.02% | -1.29% |
| **Top-10** | **75.96%** | 75.32% | -0.64% |
| **Top-50** | **93.59%** | 93.27% | -0.32% |

**Conclusioni:** Lieve calo. Nella VPR urbana, la geometria laterale (destra/sinistra) è una feature discriminante strutturale che non va alterata. Tecnica disabilitata.

---

## 5. Analisi Data Augmentation: Random Resized Crop

Come ulteriore tentativo geometrico, si è introdotto il *Random Resized Crop* (RRC), che simula variazioni di scala (zoom) e di rapporto d'aspetto.

### Ipotesi
L'idea era rendere la rete robusta a variazioni di distanza e scala, insegnando che un luogo rimane identico anche se visto più da vicino (zoom-in) o con deformazioni prospettiche lievi.

### Risultati

| Metrica | Baseline (%) | Resized Crop (Scale 0.4-1.0) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 44.55% | -1.92% |
| **Top-5** | **67.31%** | 64.74% | -2.57% |
| **Top-10** | **75.96%** | 74.36% | -1.60% |
| **Top-50** | **93.59%** | 92.95% | -0.64% |

**Conclusioni:** La perdita del contesto globale dell'immagine impatta negativamente sulla capacità di riconoscere il luogo. Tecnica disabilitata.

---

## 6. Analisi Data Augmentation: Random Erasing

Per simulare le occlusioni tipiche degli ambienti urbani reali (pedoni, auto, vegetazione), si è testato il *Random Erasing*, che sovrascrive rettangoli casuali dell'immagine con valori nulli (nero).

### Ipotesi
L'obiettivo era forzare la rete a riconoscere i luoghi basandosi su porzioni parziali della scena, aumentandone la robustezza contro gli ostacoli visivi presenti nel dataset di test (GSV).

### Risultati
| Metrica | Baseline (%) | Random Erasing ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 44.23% | -2.24% |
| **Top-5** | **67.31%** | 65.38% | -1.93% |
| **Top-10** | **75.96%** | 74.68% | -1.28% |
| **Top-50** | **93.59%** | 91.67% | -1.92% |

**Conclusioni:** La rete necessita di visuali pulite per ancorare le feature sintetiche a quelle reali; le occlusioni artificiali introducono solo rumore. Tecnica disabilitata.

---

## 7. Analisi Data Augmentation: Color Jitter (Photometric Distortions)

Si è testata la manipolazione fotometrica tramite *Color Jitter* (luminosità, contrasto, saturazione), mantenendo parametri conservativi per evitare distorsioni irrealistiche.

### Configurazione Esperimento
* **Brightness, Contrast, Saturation:** 0.3 ($\pm$30%).
* **Hue:** 0.05 (variazione minima).
* **Impatto Computazionale:** Rallentamento significativo del training (CPU bottleneck).

### Risultati
| Metrica | Baseline (%) | Color Jitter (Soft) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 45.51% | -0.96% |
| **Top-5** | **67.31%** | 65.71% | -1.60% |
| **Top-10** | **75.96%** | 75.32% | -0.64% |
| **Top-50** | **93.59%** | 92.95% | -0.64% |

**Conclusioni:** Anche l'alterazione conservativa dei colori ha portato a un calo delle performance. Tecnica disabilitata.

---

## 8. Analisi Data Augmentation: Random Rotation

Come variante geometrica rispetto al *Flip*, si è testata la *Random Rotation* per simulare lievi inclinazioni della fotocamera o pendenze stradali, situazioni comuni nelle acquisizioni reali rispetto alla perfezione geometrica del render.

### Configurazione Esperimento
* **Rotazione:** $\pm$10 gradi (bilineare).
* **Probabilità ($p$):** 0.5.
* **Riempimento bordi:** Nero.

### Risultati

| Metrica | Baseline (%) | Rotation ($\pm$10°) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 43.59% | -2.88% |
| **Top-5** | **67.31%** | 64.74% | -2.57% |
| **Top-10** | **75.96%** | 75.32% | -0.64% |
| **Top-50** | **93.59%** | 92.31% | -1.28% |

**Conclusioni:** L'allineamento verticale degli edifici è un prior geometrico fondamentale. Romperlo degrada significativamente la precisione. Tecnica disabilitata.

---

## 9. Analisi Data Augmentation: Additive Gaussian Noise

L'ultimo esperimento di questa fase ha introdotto il *Rumore Gaussiano Additivo* per simulare la grana digitale (ISO noise) tipica dei sensori fotografici reali, assente nei render "puliti" di GTA.

### Configurazione Esperimento
* **Noise Type:** Additive Gaussian.
* **Probabilità ($p$):** 0.5.
* **Intensità (std):** 0.05 (5% del range dinamico).

### Risultati

| Metrica | Baseline (%) | Gaussian Noise ($p=0.5$) | Delta |
| :--- | :---: | :---: | :---: |
| **Top-1** | **46.47%** | 44.87% | -1.60% |
| **Top-5** | **67.31%** | 65.71% | -1.60% |
| **Top-10** | **75.96%** | **75.96%** | **0.00%** |
| **Top-50** | **93.59%** | 92.63% | -0.96% |

**Conclusioni:** Sebbene sia stata la tecnica meno distruttiva (la Top-10 è rimasta invariata), ha comunque causato una regressione sulla Top-1. La tecnica è stata disabilitata.

---

## 10. Analisi Qualitativa degli Errori (Visual Inspection)

Prima di concludere l'esperimento, e su suggerimento del supervisore, è stata condotta un'analisi visiva dei "Top-20 Worst Errors" commessi dal modello Baseline (casi in cui la confidenza verso il falso positivo era massimale rispetto al ground truth).

L'ispezione ha rivelato tre pattern d'errore ricorrenti non legati alla geometria semplice:
1.  **Transient Objects (Veicoli/Camion):** Quando la query è dominata da un veicolo pesante in primo piano, la rete tende a fare "Vehicle Re-Identification" piuttosto che Place Recognition, matchando altri veicoli simili nel database e ignorando lo sfondo.
2.  **Vegetazione Invasiva:** In presenza di folte chiome d'albero, la rete matcha la texture delle foglie piuttosto che la struttura architettonica retrostante (spesso visibile nel Ground Truth scattato in stagioni diverse).
3.  **Disparità di Scala:** Errori causati da un forte zoom-out (panoramica) nella query rispetto a un dettaglio (zoom-in) nel database, o viceversa.

---

## 11. Esperimento Finale: "Targeted Augmentation Cocktail"

Sulla base dell'analisi visiva, è stato progettato un ultimo test mirato per contrastare specificamente i problemi emersi, combinando le tecniche che teoricamente avrebbero dovuto mitigarli.

### Configurazione
* **Random Erasing ($p=0.5$):** Per simulare l'occlusione dei veicoli e forzare la rete a guardare i bordi non occlusi.
* **Extended Random Resized Crop (Scale 0.5 - 1.5):** Introdotto padding per permettere crop più grandi dell'immagine originale (Zoom Out) e gestire le variazioni di scala.
* **Color Jitter (Soft):** Per variare la colorazione della vegetazione.
* **Horizontal Flip:** Disabilitato definitivamente per garantire coerenza geometrica (sinistra $\neq$ destra).

Per garantire solidità statistica, l'addestramento è stato eseguito su **3 Run indipendenti** (Seed 42, 100, 1234).

### Risultati (Media su 3 Run)

| Metrica | Baseline (Run 0) | Cocktail (Avg 3 Run) | Delta |
| :--- | :---: | :---: | :---: |
| **Recall@1** | **46.47%** | 45.72% | -0.75% |
| **Recall@5** | **67.31%** | 66.67% | -0.64% |
| **Recall@10** | **75.96%** | 75.64% | -0.32% |

---

## Conclusioni Finali Esperimento 003

L'esperimento 003 si conclude con un risultato controintuitivo ma netto: **nessuna strategia di preprocessing o data augmentation ha superato la Baseline "pulita" (solo Resize)**.

L'analisi dimostra che:
1.  Il **Domain Gap** tra GTA e GSV è talmente ampio a livello semantico (render vs reale) che le augmentation classiche introducono più rumore che robustezza.
2.  La **Coerenza Geometrica** perfetta di GTA è l'asset principale per l'apprendimento; alterarla (con crop spinti, rotazioni o occlusioni come l'Erasing) degrada la capacità del modello di estrarre feature discriminative.
3.  La backbone attuale (**ResNet50**) appare satura e incapace di discriminare semanticamente oggetti mobili (camion) da strutture fisse senza supervisione specifica.

**Next Steps:**
Alla luce di questi risultati, si abbandona l'ottimizzazione del preprocessing per procedere con l'**Esperimento 004**, focalizzato sul cambio di architettura verso modelli a più alta capacità semantica e pre-addestrati con tecniche di Self-Supervised Learning.