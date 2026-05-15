# Reconocimiento inteligente de modulaciones digitales con DSP e IA

Proyecto de investigacion para clasificar modulaciones digitales usando senales IQ sinteticas, extraccion de rasgos DSP y un modelo de IA ligero implementado en NumPy.

## Idea central

El sistema genera senales digitales simuladas bajo distintas condiciones de canal, ruido y desplazamiento de frecuencia. Luego extrae rasgos de la senal y entrena un clasificador neuronal para reconocer el tipo de modulacion.

Modulaciones incluidas:

- BPSK
- QPSK
- 8PSK
- 16QAM
- 2FSK
- 4ASK

## Pregunta de investigacion

Que tan bien puede un modelo de IA ligero clasificar modulaciones digitales usando rasgos DSP compactos, aun cuando las senales presentan ruido, offset de frecuencia, fase aleatoria y desvanecimiento?

## Hipotesis

Un clasificador entrenado con rasgos como energia espectral, estadisticas de amplitud/fase, entropia espectral y momentos de constelacion puede reconocer modulaciones digitales con buena precision sin necesitar hardware de radio ni datasets externos.

## Estructura

```text
proyecto_dsp_modulaciones
|-- README.md
|-- propuesta_investigacion.md
|-- marco_teorico_y_justificacion.md
|-- notebook_proyecto_modulaciones.ipynb
|-- requirements.txt
|-- outputs/
|   |-- summary.json
|   |-- confusion_matrix.csv
|   |-- research/
|   `-- notebook_assets/
|-- src/
|   |-- dsp_features.py
|   |-- experiment.py
|   |-- predict_demo.py
|   |-- research_experiments.py
|   |-- sanity_checks.py
|   |-- signal_generator.py
|   `-- simple_mlp.py
```

## Instalacion

Desde la carpeta raiz del proyecto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Si estas usando el Python incluido por Codex, tambien puedes ejecutar los scripts directamente con esa ruta.

## Verificacion rapida

```powershell
python src\sanity_checks.py
```

Este chequeo valida que el generador de senales, la extraccion de rasgos y el MLP funcionen con una corrida pequena.

## Experimento base

```powershell
python src\experiment.py
```

Guarda:

- `outputs/summary.json`
- `outputs/confusion_matrix.csv`
- `outputs/model_parameters.npz`

Resultado inicial reproducido:

- Accuracy de entrenamiento: 99.4%
- Accuracy de prueba: 97.9%

La mayor confusion aparece entre QPSK y 8PSK, algo esperable porque ambas modulaciones se diferencian principalmente por resolucion angular de fase.

## Experimentos de investigacion

```powershell
python src\research_experiments.py
```

Estos experimentos fortalecen la defensa del proyecto:

- IA vs baseline DSP de distancia a centroides.
- Validacion con multiples semillas.
- Robustez por SNR.
- Ablation study por grupos de features: amplitud, IQ, fase, espectro y todos.
- Generalizacion a canales no ideales: AWGN, Rayleigh y multipath.

Archivos generados:

```text
outputs/research/baseline_vs_ai.csv
outputs/research/repeated_trials.csv
outputs/research/snr_sweep.csv
outputs/research/ablation_features.csv
outputs/research/channel_generalization.csv
outputs/research/research_summary.json
```

Resultados principales:

- Baseline DSP: 82.1%
- MLP con features DSP: 98.0%
- El grupo de features de fase fue el mas informativo por separado.
- El canal multipath fue el escenario mas dificil para generalizacion.

## Notebook principal y documento

El proyecto usa un solo notebook principal:

```text
notebook_proyecto_modulaciones.ipynb
```

Este notebook unifica la explicacion del problema, las senales IQ, los rasgos DSP, el modelo MLP, las metricas, la validacion multisemilla, robustez por SNR, ablation study y generalizacion de canal. En la entrega publica se mantiene un solo notebook principal para evitar duplicidad.

El documento fuente es `marco_teorico_y_justificacion.md`.

Los scripts generadores usados para construir notebooks y documentos se mantienen solo en la carpeta local de trabajo y no forman parte de la entrega publica del repositorio.
