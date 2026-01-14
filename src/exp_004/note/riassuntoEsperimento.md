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

