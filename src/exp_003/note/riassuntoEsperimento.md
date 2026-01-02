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
