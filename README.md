# Reconocimiento inteligente de modulaciones digitales con DSP e IA

Proyecto de investigacion orientado a clasificar modulaciones digitales usando senales IQ sinteticas, extraccion de rasgos DSP y un modelo de IA ligero implementado en NumPy.

## Idea central

El sistema genera senales digitales simuladas bajo distintas condiciones de canal, ruido y desplazamiento de frecuencia. Luego extrae rasgos de la senal y entrena un clasificador neuronal para reconocer el tipo de modulacion.

Modulaciones iniciales:

- BPSK
- QPSK
- 8PSK
- 16QAM
- 2FSK
- 4ASK

## Pregunta de investigacion

¿Que tan bien puede un modelo de IA ligero clasificar modulaciones digitales usando rasgos DSP compactos, aun cuando las senales presentan ruido, offset de frecuencia, fase aleatoria y desvanecimiento?

## Hipotesis

Un clasificador entrenado con rasgos como energia espectral, estadisticas de amplitud/fase, entropia espectral y momentos de constelacion puede reconocer modulaciones digitales con buena precision sin necesitar hardware de radio ni datasets externos.

## Estructura

```text
proyecto_dsp_modulaciones
├── README.md
├── propuesta_investigacion.md
├── requirements.txt
├── outputs
└── src
    ├── dsp_features.py
    ├── experiment.py
    ├── signal_generator.py
    └── simple_mlp.py
```

## Como ejecutar

Desde la carpeta de trabajo:

```powershell
& 'C:\Users\josaf\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\proyecto_dsp_modulaciones\src\experiment.py'
```

El script entrena el modelo y guarda resultados en:

```text
proyecto_dsp_modulaciones/outputs
```

## Resultados esperados

- Accuracy general del clasificador.
- Matriz de confusion por modulacion.
- Ranking de modulaciones mas faciles y mas dificiles.
- Archivo CSV con resultados.
- Base para extender a espectrogramas, CNN o Transformers si luego instalas PyTorch.

## Resultado inicial obtenido

Con 1,920 senales sinteticas y una division 75/25 entrenamiento/prueba, el experimento inicial obtuvo:

- Accuracy de entrenamiento: 99.4%
- Accuracy de prueba: 97.9%

La mayor confusion aparece entre QPSK y 8PSK, que es un resultado esperable porque ambas modulaciones se diferencian principalmente por la resolucion angular de la fase.

## Extension investigativa

Para que el proyecto no sea solo una demostracion de accuracy, se agregaron experimentos de investigacion:

- IA vs baseline DSP de distancia a centroides.
- Robustez por SNR.
- Ablation study por grupos de features: amplitud, IQ, fase, espectro y todos.
- Generalizacion a canales no ideales: AWGN, Rayleigh y multipath.

Ejecutar:

```powershell
& 'C:\Users\josaf\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' '.\proyecto_dsp_modulaciones\src\research_experiments.py'
```

Resultados principales:

- Baseline DSP: 82.1%
- MLP con features DSP: 98.0%
- El grupo de features de fase fue el mas informativo por separado.
- El canal multipath fue el escenario mas dificil para generalizacion.

Archivos generados:

```text
outputs/research/baseline_vs_ai.csv
outputs/research/snr_sweep.csv
outputs/research/ablation_features.csv
outputs/research/channel_generalization.csv
outputs/research/research_summary.json
```
