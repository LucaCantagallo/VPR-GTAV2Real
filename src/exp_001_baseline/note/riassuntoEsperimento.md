## ESPERIMENTO 0 - baseline

## Setup
- **Dataset train:** immagini urbane sintetiche (GTA)
- **Dataset test:** immagini urbane reali (Tokyo247)
- **Loss:** TripletMarginWithDistanceLoss (basata su cosine similarity)
- **Batch size:** 32
- **Learning rate:** 1e-4
- **N epochs:** variabile (da 1 a 400)
- **Valutazione:** Recall@1 (proxy dell’accuracy del modello)

---

## Risultati principali

| Pretraining | Epoch 1 | Epoch 400 | Trend | Osservazioni |
|--------------|----------|-----------|--------|---------------|
| **ImageNet** | ≈ 46% | ≈ 48% | 🔼 Leggero miglioramento | Performance stabile, mantiene la capacità di generalizzare |
| **Places365** | ≈ 35% | ≈ 8% | 🔻 Collasso drastico | Fortissimo overfitting al dominio GTA |

---

## Considerazioni principali

### 1. ImageNet regge nel tempo
- Le feature di **ImageNet** sono di tipo **low-level**: bordi, texture, forme, gradienti.
- Queste caratteristiche non sono legate a un contesto semantico specifico e restano **robuste al cambio di dominio**.
- Anche dopo molte epoche, il modello **non overfit** sul dominio sintetico.
- La performance resta stabile o cresce leggermente → segno che il modello affina solo la proiezione, non le feature base.

**Conclusione:** le feature di ImageNet sono “neutre” rispetto al dominio e generalizzano bene anche da sintetico → reale.

---

### 2. Places365 crolla con l’addestramento
- Places365 è allenato su **scene reali e semantiche** (cucina, strada urbana, spiaggia, ecc.).
- Le feature sono di **alto livello**, fortemente dipendenti dal contesto visivo e dall’aspetto reale.
- Quando viene riaddestrato su GTA (che ha texture, illuminazioni e strutture sintetiche), il modello:
  - **sovrascrive** le feature reali con quelle del dominio sintetico;
  - **perde la capacità di riconoscere scene reali**;
  - e mostra un chiaro caso di **catastrophic forgetting**.

 **Conclusione:** Places365 è utile *solo* se il dominio di training e test sono simili (entrambi reali).  
Nel trasferimento sintetico → reale, la semantica diventa un punto debole.

---

### 3. Differenze strutturali tra i due pretraining

| Aspetto | **ImageNet** | **Places365** |
|----------|---------------|----------------|
| Tipo di feature | Low-level, oggettuali | High-level, semantiche |
| Dipendenza dal dominio | Bassa | Alta |
| Robustezza cross-domain | Alta | Bassa |
| Rischio di overfitting su GTA | Basso | Molto alto |
| Generalizzazione su reale | Buona | Drammatica perdita |
| Evoluzione con le epoche | Stabile / miglioramento | Degradazione rapida |

---

### 4. Interpretazione intuitiva
Quando il modello impara su GTA:

- Con **ImageNet**, “vede” solo pattern visivi e non si cura del dominio → continua a riconoscere strutture.
- Con **Places365**, cerca di imparare scene coerenti con il dominio di training → GTA lo inganna, e “disimpara” le scene reali.

È come se Places365 fosse troppo intelligente: prova a capire *cosa* sta vedendo, ma nel dominio sintetico quel “cosa” non esiste.  
ImageNet invece si limita a confrontare *forme* e *texture* — ed è proprio ciò che serve nel VPR cross-domain.

---

### 5. Cosa ne traggo

1. **Per il transfer sintetico → reale, feature più generiche vincono.**  
   → ImageNet resta la base più robusta.

2. **Places365 è ottimo solo se il training è su dati reali.**  
   → Se GTA è nel loop, meglio evitarlo o congelare la backbone.

3. **La semantica non sempre aiuta:** in compiti di matching visivo, conta più la *consistenza geometrica* che il significato della scena.

4. **Overtraining peggiora il transfer.**  
   → Più epoche = più specializzazione al dominio GTA = meno capacità di generalizzare.

---

## 6. Idee per futuri esperimenti

- Provare **feature self-supervised** (CLIP, DINOv2, SimCLR): spesso uniscono robustezza di ImageNet + semantica di Places.
- Testare **Domain Adaptation** (AdaBN, CORAL, MMD).
- **Visualizzare le feature** (PCA / t-SNE) per vedere il collasso cross-domain.


---

## TL;DR
> - **ImageNet**: resta stabile → generalizza bene → preferibile per VPR cross-domain.  
> - **Places365**: collassa → overfit → inadatto se train = sintetico e test = reale.  
> - Più epoche ≠ meglio: nel transfer learning, *“train less, generalize more.”*
