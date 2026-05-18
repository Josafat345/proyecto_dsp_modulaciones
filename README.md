# Adaptacion de dominio para reconocimiento de modulaciones digitales

Proyecto de investigacion sobre clasificacion automatica de modulaciones digitales usando senales IQ sinteticas, rasgos de procesamiento digital de senales e inteligencia artificial ligera. La version actual no se limita a clasificar modulaciones en un escenario ideal: estudia que ocurre cuando un modelo entrenado en un dominio sintetico controlado se evalua en condiciones objetivo mas realistas.

## Idea central

El proyecto genera senales IQ por codigo, extrae rasgos DSP interpretables y entrena un clasificador MLP implementado en NumPy. Ese clasificador primero se valida en condiciones controladas y luego se somete a cambio de dominio: menor SNR, mayor offset de frecuencia, fading Rayleigh y canal multipath.

La pregunta fuerte ya no es solamente si el modelo clasifica modulaciones, sino si puede sostener su rendimiento cuando el entorno de recepcion cambia.

Modulaciones incluidas:

- BPSK
- QPSK
- 8PSK
- 16QAM
- 2FSK
- 4ASK

## Pregunta de investigacion

Como se degrada un clasificador de modulaciones entrenado con senales IQ sinteticas de dominio fuente cuando se evalua en dominios objetivo mas realistas, y que tecnicas de adaptacion pueden reducir esa degradacion?

## Hipotesis

Un modelo entrenado solo en senales AWGN de condiciones controladas tendra alta precision dentro de su dominio fuente, pero su rendimiento caera al evaluarse en senales con bajo SNR, mayor offset, fading o multipath. Esa caida puede reducirse progresivamente mediante normalizacion del dominio objetivo, alineamiento estadistico tipo CORAL, fine-tuning con pocas muestras etiquetadas y validacion con datos reales o mas cercanos a reales.

## Estructura

```text
proyecto_dsp_modulaciones
|-- README.md
|-- propuesta_investigacion.md
|-- marco_teorico_y_justificacion.md
|-- notebook_proyecto_modulaciones.ipynb
|-- requirements.txt
|-- outputs/
|   |-- baseline/
|   |   |-- summary.json
|   |   |-- confusion_matrix.csv
|   |   `-- model_parameters.npz
|   |-- research/
|   |-- domain_adaptation/
|   `-- notebook_assets/
|-- src/
|   |-- signal_generator.py
|   |-- dsp_features.py
|   |-- simple_mlp.py
|   |-- experiment.py
|   |-- predict_demo.py
|   |-- research_experiments.py
|   |-- domain_adaptation_experiments.py
|   `-- sanity_checks.py
```

## Instalacion

Desde la carpeta raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Verificacion rapida

```powershell
python src\sanity_checks.py
```

Este chequeo valida que el generador de senales, la extraccion de rasgos y el MLP funcionen con una corrida pequena.

## Experimento base

```powershell
python src\experiment.py
```

Entrena un clasificador MLP en un escenario sintetico controlado y guarda:

```text
outputs/baseline/summary.json
outputs/baseline/confusion_matrix.csv
outputs/baseline/model_parameters.npz
```

Este experimento sirve como referencia historica. Demuestra que el pipeline DSP + IA puede clasificar modulaciones, pero no es el aporte principal de la investigacion final.

## Experimentos de soporte

```powershell
python src\research_experiments.py
```

Estos experimentos respaldan la base tecnica:

- Comparacion contra un baseline DSP de distancia a centroides.
- Validacion con multiples semillas.
- Robustez ante SNR.
- Ablation study por grupos de rasgos.
- Generalizacion inicial a canales AWGN, Rayleigh y multipath.

Archivos generados:

```text
outputs/research/baseline_vs_ai.csv
outputs/research/repeated_trials.csv
outputs/research/snr_sweep.csv
outputs/research/ablation_features.csv
outputs/research/channel_generalization.csv
outputs/research/research_summary.json
```

Resultados de soporte ya obtenidos:

- Baseline DSP: 82.1%
- MLP con rasgos DSP: 98.0%
- MLP multisemilla: 98.28% +/- 0.86%
- El grupo de rasgos de fase fue el mas informativo por separado.
- El canal multipath fue el escenario mas dificil de generalizacion.

## Experimento principal: cambio de dominio

```powershell
python src\domain_adaptation_experiments.py
```

Este experimento entrena el modelo solo en un dominio fuente AWGN y lo prueba en dominios objetivo mas dificiles:

- AWGN con menor SNR y mayor offset.
- Rayleigh.
- Multipath.
- Multipath severo con bajo SNR y offset alto.

Archivos generados:

```text
outputs/domain_adaptation/domain_shift_baseline.csv
outputs/domain_adaptation/domain_shift_summary.json
```

Resultados iniciales:

```text
Fuente AWGN controlada:        100.0%
AWGN bajo SNR/offset:           82.9%
Rayleigh:                       95.7%
Multipath:                      73.3%
Multipath severo:               60.1%
```

Esta caida justifica la investigacion: el modelo no solo debe aprender modulaciones, tambien debe adaptarse a condiciones de recepcion que cambian.

## Ruta de investigacion

La investigacion queda organizada en fases:

1. Diagnostico de cambio de dominio: medir la caida sin adaptacion.
2. Normalizacion objetivo sin etiquetas: adaptar estadisticas de rasgos al dominio objetivo.
3. CORAL: alinear covarianzas entre fuente y objetivo.
4. Few-shot fine-tuning: ajustar el modelo con pocas muestras etiquetadas del dominio objetivo.
5. Validacion realista: repetir con mas semillas, escenarios severos y eventualmente RadioML o capturas SDR.

## Notebook principal

El proyecto usa un solo notebook principal:

```text
notebook_proyecto_modulaciones.ipynb
```

El notebook debe presentar primero el problema de adaptacion de dominio, luego la base DSP/IA y finalmente los experimentos que muestran la degradacion al cambiar de dominio. Los notebooks antiguos quedan archivados localmente y no forman parte del flujo principal.
