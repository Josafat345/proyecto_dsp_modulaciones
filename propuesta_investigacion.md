# Propuesta de investigacion

## Titulo tentativo

Reconocimiento inteligente de modulaciones digitales mediante procesamiento digital de senales y aprendizaje automatico ligero

## Planteamiento del problema

En sistemas de comunicaciones modernas, identificar automaticamente el esquema de modulacion de una senal puede ser util para monitoreo de espectro, radio cognitiva, diagnostico de enlaces y sistemas educativos de telecomunicaciones. Sin embargo, muchos enfoques requieren equipos de radio, datasets externos o modelos profundos pesados.

Este proyecto propone una alternativa reproducible: generar senales IQ de forma sintetica, degradarlas con efectos realistas de canal y entrenar un modelo de IA ligero usando rasgos DSP compactos.

## Objetivo general

Desarrollar y evaluar un sistema de clasificacion automatica de modulaciones digitales basado en senales IQ sinteticas, extraccion de rasgos DSP y un clasificador neuronal implementado en codigo.

## Objetivos especificos

1. Generar senales IQ para BPSK, QPSK, 8PSK, 16QAM, 2FSK y 4ASK.
2. Incorporar efectos de canal: ruido AWGN, offset de frecuencia, fase aleatoria y variacion de ganancia.
3. Extraer rasgos DSP de amplitud, fase, espectro y constelacion.
4. Entrenar un modelo de IA ligero para clasificar la modulacion.
5. Evaluar el desempeno con matriz de confusion, accuracy y analisis por clase.

## Metodologia

1. Simulacion de datos:
   - Se generan secuencias aleatorias de simbolos.
   - Se mapean a constelaciones digitales.
   - Se sobremuestrean y se pasan por un canal simulado.

2. Procesamiento DSP:
   - Normalizacion de energia.
   - Calculo de estadisticas de amplitud y fase.
   - FFT para obtener rasgos espectrales.
   - Calculo de entropia espectral, ancho de banda efectivo y relacion pico/promedio.

3. Modelo de IA:
   - Red neuronal multicapa sencilla implementada en NumPy.
   - Entrada: vector de rasgos DSP.
   - Salida: probabilidad por tipo de modulacion.

4. Evaluacion:
   - Division entrenamiento/prueba.
   - Accuracy global.
   - Matriz de confusion.
   - Analisis de errores por modulacion.
   - Repeticion con multiples semillas para estimar estabilidad del resultado.

5. Experimentos de investigacion:
   - Comparacion IA vs baseline DSP por distancia a centroides.
   - Validacion multisemilla para evitar depender de una unica corrida favorable.
   - Barrido de SNR para medir robustez ante ruido.
   - Ablation study para estimar el aporte de grupos de rasgos.
   - Prueba de generalizacion entrenando en AWGN y evaluando en AWGN, Rayleigh y multipath.

## Innovacion

El proyecto combina simulacion completa por codigo, DSP interpretable e IA ligera. No depende de hardware SDR ni datasets externos. Ademas, permite explicar las decisiones del modelo a partir de rasgos fisicos de la senal.

## Diseno experimental fortalecido

La version extendida no se limita a reportar accuracy. Se proponen cinco preguntas medibles:

1. ¿La IA mejora frente a un baseline DSP clasico?
2. ¿El resultado se mantiene estable al repetir el experimento con distintas semillas?
3. ¿A partir de que SNR el sistema se vuelve confiable?
4. ¿Que grupos de rasgos aportan mas informacion?
5. ¿El modelo generaliza cuando el canal cambia de AWGN a Rayleigh o multipath?

Resultados iniciales:

- El MLP con rasgos DSP alcanzo 98.0% de accuracy frente a 82.1% del baseline de distancia a centroides.
- La validacion con multiples semillas reporta media y desviacion estandar para que el resultado sea mas defendible.
- En el barrido por SNR, el modelo mejora claramente a partir de 5 dB y se vuelve casi perfecto desde 10 dB en el escenario simulado.
- En el ablation study, los rasgos de fase fueron los mas fuertes de forma aislada.
- Multipath fue el escenario mas dificil, bajando el accuracy del MLP a 84.6%, lo cual abre una linea de mejora realista.

## Posibles extensiones

- Usar espectrogramas como imagenes y entrenar una CNN.
- Comparar el MLP contra un clasificador basado en reglas.
- Evaluar robustez variando SNR.
- Incorporar canales Rayleigh/Rician.
- Crear una interfaz para visualizar senal temporal, FFT, constelacion y prediccion.
- Integrar un modelo tipo Audio Spectrogram Transformer si se instala PyTorch/Hugging Face.

## Referencias base

- Dobre, O. A. et al. Survey of automatic modulation classification techniques: classical approaches and new trends.
- Harper, C. A., Thornton, M. A. and Larson, E. C. Automatic Modulation Classification with Deep Neural Networks, Electronics, 2023.
- Gong, Y. et al. AST: Audio Spectrogram Transformer, Interspeech 2021.
- PyTorch Audio documentation: torchaudio pipelines and audio processing utilities.
