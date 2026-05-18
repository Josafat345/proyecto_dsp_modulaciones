# Propuesta de investigacion

## Titulo tentativo

Adaptacion de dominio para reconocimiento automatico de modulaciones digitales usando senales IQ sinteticas, rasgos DSP e inteligencia artificial ligera

## Planteamiento del problema

La clasificacion automatica de modulaciones permite que un receptor identifique el esquema de transmision de una senal sin conocerlo previamente. Esta capacidad es relevante en radio cognitiva, monitoreo de espectro, diagnostico de enlaces, sistemas de comunicacion adaptativos y ambientes educativos de telecomunicaciones.

El problema es que muchos modelos se entrenan y evaluan bajo condiciones parecidas. En esos casos, el accuracy puede ser alto, pero el resultado no necesariamente indica que el sistema funcionara cuando cambien las condiciones de recepcion. En un entorno mas realista, una senal puede llegar con menor SNR, offset de frecuencia, fase aleatoria, desvanecimiento, multipath, variacion de ganancia o distorsiones del receptor.

Por eso, el reto investigativo de este proyecto no es solamente clasificar modulaciones. El reto es estudiar la brecha entre un dominio fuente sintetico controlado y un dominio objetivo mas dificil, y evaluar estrategias para reducir la perdida de rendimiento cuando el modelo se mueve entre ambos dominios.

## Pregunta de investigacion

Como se degrada un clasificador de modulaciones entrenado con senales IQ sinteticas de dominio fuente al evaluarse en dominios objetivo mas realistas, y que tecnicas de adaptacion de dominio pueden reducir esa degradacion?

## Hipotesis

Un clasificador MLP entrenado sobre rasgos DSP de senales AWGN controladas lograra alta precision dentro del dominio fuente, pero perdera rendimiento al evaluarse en senales con bajo SNR, mayor offset, fading y multipath. La degradacion puede reducirse progresivamente mediante normalizacion del dominio objetivo sin etiquetas, alineamiento estadistico tipo CORAL y fine-tuning con pocas muestras etiquetadas del dominio objetivo.

## Objetivo general

Desarrollar y evaluar un prototipo de adaptacion de dominio para clasificacion automatica de modulaciones digitales, usando senales IQ sinteticas, rasgos DSP interpretables y un modelo de IA ligero.

## Objetivos especificos

1. Generar senales IQ sinteticas para BPSK, QPSK, 8PSK, 16QAM, 2FSK y 4ASK.
2. Simular dominios fuente y objetivo con distintas condiciones de SNR, offset de frecuencia y canal.
3. Extraer rasgos DSP de amplitud, componentes I/Q, fase y espectro.
4. Entrenar un clasificador MLP ligero sobre el dominio fuente.
5. Medir la caida de rendimiento al evaluar en dominios objetivo.
6. Comparar el modelo contra un baseline DSP basado en distancia a centroides.
7. Implementar tecnicas progresivas de adaptacion de dominio.
8. Evaluar estabilidad con multiples semillas, escenarios de canal y metricas por clase.

## Alcance actual

El proyecto ya cuenta con una base funcional:

- Generador de senales IQ para seis modulaciones digitales.
- Canal con AWGN, Rayleigh y multipath.
- Extraccion de 36 rasgos DSP.
- Modelo MLP implementado en NumPy.
- Baseline DSP de distancia a centroides.
- Validacion multisemilla, barrido por SNR y ablation study.
- Diagnostico inicial de cambio de dominio.

Resultados iniciales del diagnostico:

```text
Fuente AWGN controlada:        100.0%
AWGN bajo SNR/offset:           82.9%
Rayleigh:                       95.7%
Multipath:                      73.3%
Multipath severo:               60.1%
```

Estos resultados muestran que el modelo no falla de manera uniforme: el cambio de canal, el bajo SNR y el multipath producen una degradacion clara. Esa degradacion es la motivacion central de la investigacion.

## Antecedentes y posicionamiento

La clasificacion automatica de modulaciones ya es un tema trabajado en comunicaciones digitales. Existen enfoques clasicos basados en rasgos, enfoques basados en verosimilitud y propuestas recientes con redes profundas sobre senales IQ. Tambien existen trabajos de transferencia y adaptacion de dominio aplicados a reconocimiento de modulaciones.

Por esa razon, esta propuesta no debe presentarse como "el primer clasificador de modulaciones con IA". La vision mas defendible es otra: construir un prototipo ligero, interpretable y reproducible que mida la brecha entre entrenamiento sintetico y evaluacion en condiciones objetivo mas realistas, y que compare tecnicas simples de adaptacion antes de recurrir a modelos profundos pesados.

El aporte diferencial se ubica en cuatro puntos:

1. **Interpretabilidad:** el modelo usa rasgos DSP medibles, no solo muestras IQ crudas.
2. **Bajo costo:** el clasificador esta implementado en NumPy y puede ejecutarse sin GPU.
3. **Ruta progresiva:** se evalua la degradacion sin adaptacion y luego se proponen mejoras por etapas.
4. **Valor academico:** el proyecto permite explicar comunicacion digital, canal, SNR, features, modelo y adaptacion de dominio dentro de un mismo flujo reproducible.

La investigacion queda mejor posicionada como un estudio de **recuperacion de rendimiento ante cambio de dominio**, no como una competencia directa contra arquitecturas profundas de estado del arte.

## Metodologia propuesta

### Fase 1: Diagnostico de cambio de dominio

Se entrena el modelo en un dominio fuente AWGN con SNR relativamente alto y offset moderado. Luego se evalua en varios dominios objetivo mas dificiles. Esta fase mide la brecha inicial sin adaptacion.

### Fase 2: Normalizacion objetivo sin etiquetas

Se asume que existen senales del dominio objetivo, pero no sus etiquetas. Se calculan estadisticas del dominio objetivo y se usan para normalizar los rasgos antes de clasificarlos. Esta fase evalua si una adaptacion simple de estadisticas reduce la caida.

### Fase 3: Alineamiento CORAL

Se aplica CORAL para alinear la covarianza de los rasgos fuente con la covarianza del dominio objetivo. Esto busca que las distribuciones de features sean mas compatibles sin requerir etiquetas del dominio objetivo.

### Fase 4: Fine-tuning con pocas muestras

Se simula un caso donde el investigador tiene pocas senales etiquetadas del dominio objetivo. El modelo se ajusta con esas muestras y se mide si mejora sin sobreajustarse.

### Fase 5: Validacion extendida

Se repiten los experimentos con diferentes semillas, rangos de SNR, canales y matrices de confusion. Como extension fuerte, el dominio objetivo puede reemplazarse por RadioML o por capturas SDR reales.

## Variables y metricas

Variables independientes:

- Tipo de canal: AWGN, Rayleigh, multipath y posibles extensiones Rician.
- Rango de SNR.
- Offset de frecuencia.
- Estrategia de adaptacion: ninguna, normalizacion objetivo, CORAL, fine-tuning.
- Cantidad de muestras etiquetadas del dominio objetivo.

Metricas:

- Accuracy global.
- Caida de accuracy respecto al dominio fuente.
- Accuracy por clase.
- Matriz de confusion.
- Media y desviacion estandar con multiples semillas.
- Diferencia entre MLP y baseline DSP.

## Innovacion

El valor del proyecto esta en construir una ruta reproducible entre simulacion y operacion realista. La investigacion combina tres ideas:

1. Senales IQ sinteticas controlables por codigo.
2. Rasgos DSP interpretables que permiten explicar el comportamiento del modelo.
3. Adaptacion de dominio para estudiar la brecha entre entrenamiento sintetico y prueba realista.

Esto diferencia el trabajo de una clasificacion comun de modulaciones. El aporte no es decir que un MLP clasifica BPSK o QPSK, sino analizar cuando deja de funcionar, por que ocurre y que estrategias reducen esa perdida.

## Entregables esperados

- Notebook principal con explicacion teorica, metodologia, graficas y resultados.
- Scripts reproducibles en `src/`.
- Resultados CSV/JSON organizados por fase experimental.
- Documento de marco teorico y justificacion.
- Comparacion clara entre baseline DSP, MLP sin adaptacion y MLP con adaptacion.

## Referencias base

- Dobre, O. A. et al. Survey of automatic modulation classification techniques: classical approaches and new trends.
- O'Shea, T. y West, N. RadioML 2016.10a Dataset.
- Sun, B., Feng, J. y Saenko, K. Return of Frustratingly Easy Domain Adaptation.
- Harper, C. A., Thornton, M. A. y Larson, E. C. Automatic Modulation Classification with Deep Neural Networks.
