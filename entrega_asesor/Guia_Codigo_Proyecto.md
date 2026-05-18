# Guia para explicar el codigo al asesor

## Idea general del codigo

El codigo esta organizado como un pipeline de comunicaciones e inteligencia artificial:

```text
senal IQ sintetica
-> canal simulado
-> extraccion de rasgos DSP
-> clasificador MLP
-> evaluacion base
-> diagnostico de cambio de dominio
```

La idea importante para explicar es que el modelo no recibe directamente una explicacion teorica de la modulacion. Recibe mediciones numericas de la senal: amplitud, fase, componentes I/Q y espectro. Luego aprende a asociar esos rasgos con cada tipo de modulacion.

## Orden recomendado para mostrar los archivos

### 1. `src/signal_generator.py`

Este archivo crea las senales de prueba.

Que decir:

- Aqui se generan las modulaciones BPSK, QPSK, 8PSK, 16QAM, 2FSK y 4ASK.
- Primero se crea una senal ideal.
- Luego se le aplica un canal simulado con ruido, fase aleatoria, offset de frecuencia, Rayleigh o multipath.
- Esto permite controlar el experimento sin depender inicialmente de hardware SDR o datasets externos.

Funciones clave:

- `generate_clean_signal()`: crea la senal ideal.
- `apply_channel()`: agrega condiciones realistas.
- `generate_dataset()`: crea un conjunto balanceado de senales y etiquetas.

### 2. `src/dsp_features.py`

Este archivo convierte cada senal IQ en un vector de rasgos.

Que decir:

- La senal completa puede tener muchas muestras, pero el modelo recibe 36 rasgos compactos.
- Los rasgos se agrupan en amplitud, I/Q, fase y espectro.
- Esto hace el modelo mas interpretable, porque se puede explicar que mide cada grupo.

Funciones clave:

- `extract_features()`: extrae los 36 rasgos de una senal.
- `extract_feature_matrix()`: aplica la extraccion a muchas senales.
- `standardize_train_test()`: normaliza los rasgos usando estadisticas del entrenamiento.

### 3. `src/simple_mlp.py`

Este archivo contiene el modelo de IA.

Que decir:

- Es una red neuronal pequena implementada solo con NumPy.
- No depende de PyTorch ni TensorFlow.
- Tiene una capa oculta, ReLU, softmax y entrenamiento con gradiente.
- La salida es una probabilidad por modulacion.

Funciones clave:

- `SimpleMLP.fit()`: entrena el modelo.
- `SimpleMLP.predict_proba()`: devuelve probabilidades.
- `SimpleMLP.predict()`: devuelve la clase con mayor probabilidad.
- `accuracy()`: calcula porcentaje de aciertos.
- `confusion_matrix()`: muestra errores por clase.

### 4. `src/experiment.py`

Este es el experimento base.

Que decir:

- Genera un dataset sintetico.
- Extrae rasgos DSP.
- Entrena el MLP.
- Guarda matriz de confusion, resumen y parametros.
- Sirve como prueba de que el pipeline completo funciona.

Archivos que produce:

- `outputs/baseline/summary.json`
- `outputs/baseline/confusion_matrix.csv`
- `outputs/baseline/model_parameters.npz`

### 5. `src/research_experiments.py`

Este archivo fortalece la parte investigativa.

Que decir:

- No se queda solo en un accuracy.
- Compara contra un baseline DSP.
- Repite con varias semillas.
- Evalua robustez por SNR.
- Hace ablation study para ver que rasgos aportan mas.
- Evalua generalizacion a otros canales.

Resultados importantes:

- Baseline DSP: 82.1%.
- MLP con rasgos DSP: 98.0%.
- MLP multisemilla: 98.28% +/- 0.86%.
- Multipath aparece como escenario dificil.

### 6. `src/domain_adaptation_experiments.py`

Este es el archivo mas importante para el nuevo enfoque.

Que decir:

- Entrena el modelo en un dominio fuente AWGN controlado.
- Luego lo prueba en dominios objetivo mas dificiles.
- No adapta todavia el modelo; solo mide la caida.
- Esta es la Fase 1 de la investigacion.

Resultado clave:

```text
Fuente AWGN controlada:        100.0%
AWGN bajo SNR/offset:           82.9%
Rayleigh:                       95.7%
Multipath:                      73.3%
Multipath severo:               60.1%
```

Lectura:

El modelo funciona muy bien cuando el dominio de prueba se parece al dominio de entrenamiento, pero cae cuando el canal se vuelve mas dificil. Esa caida justifica implementar adaptacion de dominio.

### 7. `src/predict_demo.py`

Este archivo es una demostracion rapida.

Que decir:

- Carga el modelo entrenado por `experiment.py`.
- Genera senales nuevas.
- Muestra la prediccion y la confianza del modelo.
- Sirve para ensenar el uso practico del clasificador.

### 8. `src/sanity_checks.py`

Este archivo verifica que el proyecto no este roto.

Que decir:

- Ejecuta pruebas pequenas con `assert`.
- Comprueba que el generador, los rasgos y el MLP funcionen juntos.
- Es util antes de presentar o antes de subir cambios a GitHub.

## Frase corta para explicar el flujo

> Primero genero senales IQ sinteticas de varias modulaciones. Despues simulo condiciones de canal para que no sean ideales. Luego convierto cada senal en 36 rasgos DSP interpretables y entreno un MLP ligero. Finalmente, no solo mido accuracy base, sino que pruebo cuanto cae el modelo cuando cambia el dominio de evaluacion. Esa caida es lo que motiva la adaptacion de dominio.

## Que parte destacar ante el asesor

Lo mas importante no es decir que el modelo llega a 98% o 100%. Eso puede pasar porque el escenario sintetico es controlado.

Lo importante es mostrar que:

1. El pipeline base funciona.
2. Hay comparacion contra baseline DSP.
3. Hay evidencia de robustez y ablation study.
4. El modelo cae en dominios objetivo dificiles.
5. Esa caida abre la investigacion de adaptacion de dominio.

## Preguntas que el asesor podria hacer

### Por que no usar directamente una CNN?

Porque primero se quiere una solucion interpretable y ligera. Una CNN puede ser una extension, pero el enfoque actual permite explicar que rasgos fisicos ayudan o fallan.

### Por que usar datos sinteticos?

Porque permiten controlar modulacion, SNR, canal y offset. La limitacion se reconoce, y por eso la ruta final puede incluir RadioML o capturas SDR.

### Que aporta la adaptacion de dominio?

Aporta una forma de estudiar y reducir la diferencia entre el dominio donde se entrena el modelo y el dominio donde se quiere usar.

### Que sigue despues de esta version?

Implementar Fase 2: normalizacion del dominio objetivo sin etiquetas, y comparar si mejora sobre la Fase 1 sin adaptacion.
