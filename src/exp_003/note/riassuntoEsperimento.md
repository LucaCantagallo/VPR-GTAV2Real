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