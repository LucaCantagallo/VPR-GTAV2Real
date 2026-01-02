
## Gta2Real: Analisi e Strategie

L'utilizzo di dataset sintetici (**GTAV**) per il training di modelli di Visual Place Recognition (VPR) destinati al mondo reale introduce il problema critico del **Domain Shift**.
Le differenze statistiche tra il dominio sorgente ($\mathcal{D}_s$, sintetico) e il dominio target ($\mathcal{D}_t$, reale) sono profonde:

* **Differenze Low-level:** Texture prive di rumore sensore, illuminazione *ray-traced* perfetta, assenza di motion blur e aberrazioni cromatiche.
* **Differenze High-level:** Palette cromatica artificiale, pattern ripetitivi e layout urbani specifici del motore di gioco.

Queste discrepanze causano un overfitting sul dominio sintetico, degradando drasticamente le performance di generalizzazione su dataset di test reali come **GSV-Cities**.

L'obiettivo è implementare una pipeline di **Generative Domain Adaptation**. Si intende apprendere una funzione di mapping $G(x_s) \rightarrow \hat{x}_t$ in grado di trasformare l'immagine sintetica in una fotorealistica, rispettando due vincoli fondamentali per il VPR:
1.  **Preservazione Geometrica:** La struttura della scena (edifici, strade, layout) deve rimanere identica.
2.  **Coerenza Semantica:** Il contenuto non deve essere stravolto (es. le condizioni meteo e la tipologia di oggetti devono rimanere coerenti).

### 2.1 Analisi Qualitativa delle Tecniche

Di seguito si riportano i risultati dei tre principali approcci qualitativi tentati per risolvere il problema.

#### 2.1.1 Esperimento Analisi Qualitativa 1: Approccio GAN-based (CycleGAN)

**Razionale e Metodologia**
In prima istanza, è stato adottato l'approccio classico per il trasferimento di stile non supervisionato (*Unpaired Image-to-Image Translation*). Non disponendo di coppie di immagini identiche, la scelta è ricaduta su **CycleGAN** (Zhu et al., 2017).
Il modello si basa su due principi competitivi: **Adversarial Loss** (per il realismo) e **Cycle Consistency Loss** (per la coerenza $A \rightarrow B \rightarrow A \approx A$). Per l'esperimento sono stati utilizzati pesi pre-addestrati sul dataset urbano **Cityscapes**.

**Analisi dell'insuccesso**
L'inferenza ha prodotto risultati inaccettabili (Fig. 1). Il modello non è riuscito a convergere verso uno stile fotografico coerente, introducendo invece:
* **High-frequency noise:** Rumore visivo che distrugge i dettagli fini.
* **Checkerboard artifacts:** Artefatti a scacchiera tipici delle deconv nelle GAN mal condizionate.
* **Distorsione geometrica:** Le linee rette degli edifici sono risultate ondulate o spezzate.

![Qualitative Analysis Failure - CycleGAN Inference](reportImages/0001.png)
*Fig 1: Fallimento dell'approccio CycleGAN. L'output è dominato da rumore e artefatti.*

#### 2.1.2 Esperimento Analisi Qualitativa 2: Controlled Latent Diffusion Models

Abbandonate le GAN, la strategia è virata verso i **Latent Diffusion Models (LDM)**.

**Setup e Infrastruttura**
Per gestire i requisiti computazionali, l'ambiente è stato migrato da Cloud (Colab) a un'**infrastruttura locale** (Server con GPU dedicata). È stata implementata una rigorosa segregazione degli ambienti tramite **Conda**:
* `gta_gen`: Ambiente dedicato alla generazione (PyTorch + Diffusers + xFormers).
* `vpr`: Ambiente dedicato al training della rete di recognition.
Questa separazione è stata necessaria per risolvere conflitti critici di versione CUDA e dipendenze.

**Metodologia: Text-to-Image + ControlNet**
È stato utilizzato il modello **Stable Diffusion 1.5** caricando lo specifico checkpoint **epiCRealism**, scelto per la sua capacità superiore nel generare texture fotografiche realistiche rispetto al modello base.
Per vincolare la geometria, il modello è stato accoppiato a **ControlNet** con pre-processore **Canny Edge**.
* **Input:** Immagine GTA processata con filtro Canny per estrarre i bordi.
* **Prompt:** *"raw photo, urban street, realistic lighting, high quality"*.
* **Logica:** Si è chiesto al modello di riempire i contorni estratti da GTA ("traccia") con texture realistiche generate dal prompt.

**Analisi Qualitativa: Successo Parziale e Allucinazioni**
I risultati visivi (Fig. 2) mostrano un netto salto di qualità nelle texture, ma evidenziano un grave problema di **Semantic Gap**.

![GTA vs Real Comparison](reportImages/GTA_vs_Real_Comparison.png)
*Fig 2: Confronto GTA (Sinistra) vs SD 1.5 ControlNet (Destra).*

**Analisi Critica:**
1.  **Semantic Hallucination:** Il modello ha allucinato dettagli inesistenti. Scene diurne in GTA sono state trasformate arbitrariamente in scene notturne, piovose o al tramonto, poiché il prompt non era sufficientemente condizionato dalla palette originale.
2.  **Perdita di Identità:** Nonostante il vincolo forte di ControlNet (Canny), la percezione globale della scena viene alterata. Il modello tende a "re-immaginare" la scena invece di "tradurla".
3.  **Rischio per il VPR:** Addestrare una rete su immagini semanticamente diverse dal ground truth geometrico rischia di insegnare feature errate al modello.

#### 2.1.3 Esperimento Analisi Qualitativa 3: Fourier Domain Adaptation (FDA)

**Razionale: Spettro vs Generazione**
Per ovviare alle "allucinazioni" dei modelli generativi, si è tentato un approccio matematico deterministico: **Fourier Domain Adaptation (FDA)** (Yang et al., 2020).
L'ipotesi è che lo "stile" (illuminazione, texture globale) sia codificato nell'**Ampiezza** dello spettro di Fourier, mentre il "contenuto" (bordi, geometria) risieda nella **Fase**.
L'obiettivo era sostituire l'ampiezza delle basse frequenze dell'immagine sintetica ($\mathcal{F}^A_{src}$) con quella di un'immagine reale di riferimento ($\mathcal{F}^A_{trg}$), mantenendo intatta la fase originale ($\mathcal{F}^P_{src}$).

$$
\mathcal{F}_{out} = \mathcal{F}^{-1} ( M_{\beta}(\mathcal{F}^A_{src}, \mathcal{F}^A_{trg}) \cdot e^{i \mathcal{F}^P_{src}} )
$$

**Risultato: Artefatti Spettrali**
I risultati (Fig. 3) evidenziano criticità intrinseche al metodo spettrale "naive".

![FDA Results - Spectral Artifacts](reportImages/GTA_plus_Real_eq_FDA.png)
*Fig 3: Risultati FDA. Colonna 1: Input GTA; Colonna 2: Target Reale; Colonna 3: Output.*

**Analisi del Fallimento:**
1.  **Ringing Artifacts (Banding):** Lo scambio brusco delle ampiezze crea discontinuità che generano onde e bande colorate ("effetto arcobaleno").
2.  **Semantic Mismatch:** FDA è agnostico al contenuto spaziale. Se l'immagine reale di riferimento ha colori dominanti specifici (es. un muro beige), FDA spalma quella distribuzione spettrale globalmente, colorando l'asfalto di GTA con tinte errate.
3.  **Assenza di Texture:** FDA ricolora i pixel esistenti ma non genera texture realistiche mancanti.

### 2.2 Considerazioni Attuali

Dall'analisi qualitativa dei tre esperimenti emerge un trade-off chiaro:
* **CycleGAN:** Troppo instabile e distruttiva per la geometria.
* **FDA:** Conservativa sulla geometria ma introduce artefatti visivi e non risolve la mancanza di texture.
* **Diffusion (SD+ControlNet):** Eccellente fotorealismo ma semanticamente "creativa" e poco fedele.

La pipeline generativa pura (*Text-to-Image* guidata da bordi) concede troppa libertà al modello.
Tuttavia, i Latent Diffusion Models rimangono la strada più promettente per la qualità delle texture. La ricerca si sposterà ora verso tecniche di **Image-to-Image (Img2Img)** con denoising parziale, per vincolare non solo i bordi ma anche la palette cromatica originale.

Parallelamente, si procederà a valutare l'efficacia di tecniche di **Data Augmentation** classiche (non generative) per stabilire una baseline solida di robustezza al domain shift.