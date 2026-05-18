# Marco teorico y justificacion del proyecto

## Titulo del proyecto

**Adaptacion de dominio para reconocimiento automatico de modulaciones digitales usando senales IQ sinteticas, rasgos DSP e inteligencia artificial ligera**

## 1. Introduccion

En los sistemas modernos de comunicacion digital, la informacion se transmite modificando ciertas propiedades de una senal portadora, como su amplitud, fase o frecuencia. A este proceso se le llama **modulacion**. Dependiendo de como se alteren esas propiedades, aparecen distintos esquemas de modulacion, por ejemplo BPSK, QPSK, 8PSK, 16QAM, FSK o ASK.

La identificacion automatica de modulaciones, conocida en ingles como **Automatic Modulation Classification** (AMC), consiste en determinar que tipo de modulacion tiene una senal recibida sin conocer de antemano los datos transmitidos. Este problema es importante porque en un receptor inteligente no siempre se conoce el formato de la senal que llega. Antes de demodularla correctamente, el sistema necesita reconocer que tipo de modulacion esta observando.

En este proyecto se desarrolla un sistema que genera senales digitales sinteticas, les aplica condiciones de canal no ideales y luego usa procesamiento digital de senales (DSP) junto con un modelo de inteligencia artificial para clasificarlas. La investigacion actual va un paso mas alla: no solo busca demostrar que el clasificador funciona en datos sinteticos, sino medir que ocurre cuando el modelo entrenado en un dominio fuente controlado se prueba en dominios objetivo mas realistas.

El eje central pasa a ser la **adaptacion de dominio**. En otras palabras, se estudia la brecha entre entrenar con senales simuladas relativamente limpias y evaluar con senales afectadas por bajo SNR, mayor offset de frecuencia, fading y multipath. Esta brecha es importante porque un modelo puede obtener alto accuracy en laboratorio y aun asi degradarse cuando cambian las condiciones de recepcion.

## 2. Por que se hace esta investigacion

Esta investigacion se realiza porque la clasificacion automatica de modulaciones es un problema real dentro de las comunicaciones digitales, la radio cognitiva, el monitoreo de espectro y los receptores inteligentes. En un sistema ideal, el receptor conoce todos los parametros de transmision. En la practica, una senal puede llegar con ruido, distorsion, variaciones de fase, desplazamiento de frecuencia, desvanecimiento o multiples trayectorias. Esto vuelve dificil identificar correctamente el esquema de modulacion.

La motivacion principal es estudiar si se puede construir una solucion que combine dos mundos:

1. **DSP clasico**, que permite extraer informacion fisica de la senal.
2. **IA ligera**, que aprende patrones a partir de esos rasgos sin requerir un modelo profundo pesado.

El proyecto tambien tiene valor academico porque permite estudiar conceptos de telecomunicaciones de una forma reproducible. No se requiere radio definida por software, antenas ni datasets externos. Todas las senales se generan por codigo, lo que permite controlar el tipo de modulacion, el nivel de ruido, el canal y las condiciones del experimento.

La investigacion no se limita a entrenar un modelo y reportar accuracy. Tambien compara contra un baseline DSP, evalua robustez por SNR, estudia que grupos de rasgos aportan mas informacion y prueba generalizacion a canales no ideales. Con el nuevo enfoque, estas pruebas quedan como soporte para una pregunta mas fuerte: como reducir la caida de rendimiento cuando existe cambio de dominio entre el entrenamiento y la evaluacion.

### 2.1 Por que la adaptacion de dominio fortalece el proyecto

Una investigacion sobre clasificacion de modulaciones puede quedarse corta si solo muestra que un modelo reconoce senales generadas con el mismo simulador usado para entrenar. Ese resultado es util, pero ya ha sido explorado ampliamente en la literatura.

La adaptacion de dominio vuelve el problema mas realista porque introduce una separacion entre:

- **Dominio fuente:** datos usados para entrenar, por ejemplo senales AWGN con SNR relativamente alto.
- **Dominio objetivo:** datos donde se quiere aplicar el modelo, por ejemplo senales con bajo SNR, multipath o fading.

La pregunta ya no es "cuanto accuracy tengo en mi simulador", sino "cuanto accuracy pierdo cuando el dominio cambia y que tecnica ayuda a recuperarlo". Esta formulacion hace que el proyecto sea mas defendible porque identifica una limitacion real y propone una ruta experimental para resolverla.

## 3. Conceptos basicos necesarios

### 3.1 Procesamiento digital de senales (DSP)

El procesamiento digital de senales, o **DSP**, es el conjunto de tecnicas usadas para analizar, modificar o extraer informacion de senales representadas de forma digital. En este proyecto, una senal no se analiza directamente como una onda analogica, sino como una secuencia de muestras numericas.

Algunas operaciones DSP usadas en la investigacion son:

- Normalizacion de energia.
- Separacion de componentes I y Q.
- Analisis de amplitud y fase.
- Transformada rapida de Fourier (FFT).
- Calculo de rasgos espectrales.
- Calculo de estadisticos de la senal.

La ventaja del DSP es que permite convertir una senal compleja en informacion medible. Por ejemplo, se puede calcular si una senal concentra energia en ciertas frecuencias, si su amplitud cambia mucho, si su fase salta entre valores discretos o si su espectro se dispersa por el ruido.

### 3.2 Modulacion digital

La modulacion digital consiste en representar bits mediante simbolos que modifican una portadora. Cada simbolo puede representar uno o varios bits. Los esquemas usados en el proyecto son:

- **BPSK**: usa dos estados de fase. Cada simbolo representa 1 bit.
- **QPSK**: usa cuatro estados de fase. Cada simbolo representa 2 bits.
- **8PSK**: usa ocho estados de fase. Cada simbolo representa 3 bits.
- **16QAM**: combina variaciones de amplitud y fase. Cada simbolo representa 4 bits.
- **2FSK**: representa bits usando dos frecuencias distintas.
- **4ASK**: representa simbolos usando cuatro niveles de amplitud.

Estas modulaciones son utiles para el proyecto porque no todas se diferencian de la misma manera. Algunas dependen principalmente de la fase, otras de la amplitud y otras de la frecuencia. Esto permite evaluar que rasgos DSP son mas utiles para separar cada clase.

### 3.3 Senales IQ

En comunicaciones digitales es comun representar una senal usando dos componentes:

- **I**: componente en fase, del ingles *In-phase*.
- **Q**: componente en cuadratura, del ingles *Quadrature*.

Matematicamente, una senal IQ se puede representar como una senal compleja:

```text
s[n] = I[n] + jQ[n]
```

Esta representacion es muy util porque permite analizar amplitud y fase directamente:

```text
amplitud = |s[n]|
fase = angle(s[n])
```

En el proyecto, cada senal sintetica se genera como una secuencia IQ. Luego se extraen rasgos de I, Q, amplitud, fase y espectro.

### 3.4 Constelacion

La **constelacion** es una representacion grafica de los simbolos de una modulacion en el plano I-Q. Cada punto de la constelacion representa un simbolo posible.

Por ejemplo:

- BPSK tiene dos puntos principales.
- QPSK tiene cuatro puntos.
- 8PSK tiene ocho puntos alrededor de un circulo.
- 16QAM tiene una grilla de dieciseis puntos.

La constelacion ayuda a entender por que el problema de clasificacion es posible. Si las clases tienen patrones diferentes en el plano I-Q, un sistema puede aprender a distinguirlas. Sin embargo, cuando hay ruido o distorsion, los puntos se dispersan y la clasificacion se vuelve mas dificil.

### 3.5 Ruido AWGN

AWGN significa **Additive White Gaussian Noise**, o ruido blanco gaussiano aditivo. Es un modelo comun en comunicaciones porque representa ruido aleatorio que se suma a la senal transmitida.

En forma simple:

```text
senal_recibida = senal_transmitida + ruido
```

El ruido afecta la posicion de los puntos en la constelacion, altera la amplitud y puede confundir al clasificador. Por eso se evalua el sistema con diferentes niveles de ruido.

### 3.6 Relacion senal a ruido (SNR)

La **SNR** indica que tan fuerte es la senal en comparacion con el ruido. Se expresa normalmente en decibeles (dB).

Una SNR alta significa que la senal se observa con claridad. Una SNR baja significa que el ruido domina. En el proyecto se evalua el modelo con distintos valores de SNR:

```text
-5 dB, 0 dB, 5 dB, 10 dB, 15 dB, 20 dB, 25 dB
```

Esto permite responder una pregunta importante:

**A partir de que SNR el clasificador se vuelve confiable?**

### 3.7 Desplazamiento de fase y frecuencia

En un receptor real, la senal recibida puede no estar perfectamente sincronizada con el transmisor. Esto puede producir:

- **Offset de fase**: la constelacion rota.
- **Offset de frecuencia**: la fase cambia progresivamente con el tiempo.

Estos efectos se incluyen en la simulacion para que el problema sea mas realista. Si el modelo solo funcionara con senales ideales, no seria una investigacion robusta.

### 3.8 Canal Rayleigh

El canal Rayleigh modela situaciones donde la senal recibida se ve afectada por desvanecimiento. Esto ocurre cuando la senal llega al receptor despues de reflejarse en objetos, paredes, edificios u otros obstaculos. La amplitud y fase recibidas pueden variar de forma aleatoria.

En la investigacion, Rayleigh se usa como una condicion de prueba para observar si el modelo entrenado en AWGN todavia puede reconocer modulaciones cuando cambia el comportamiento del canal.

### 3.9 Canal multipath

El canal **multipath** ocurre cuando varias copias de la misma senal llegan al receptor con diferentes retardos. Esto puede suceder por reflexiones en el entorno. El resultado es que las copias se suman y pueden distorsionar la forma de la senal.

El multipath es especialmente importante porque puede mezclar simbolos y deformar la constelacion. En los resultados del proyecto, este canal fue el mas dificil para el modelo, lo cual tiene sentido desde el punto de vista de comunicaciones.

## 4. Clasificacion automatica de modulaciones

La clasificacion automatica de modulaciones busca identificar la modulacion de una senal recibida sin conocer los bits transmitidos. Segun la literatura, este problema es complejo porque el receptor puede desconocer parametros como potencia, frecuencia portadora, fase, sincronizacion y condiciones de canal.

Tradicionalmente, existen dos enfoques:

1. **Metodos basados en verosimilitud**, que intentan comparar estadisticamente la senal contra modelos de modulacion.
2. **Metodos basados en rasgos**, que extraen caracteristicas de la senal y luego usan un clasificador.

Este proyecto usa el segundo enfoque. Primero extrae rasgos DSP y luego clasifica usando un modelo de IA. Esta eleccion es adecuada porque hace que el sistema sea interpretable: se puede explicar que informacion se le entrega al modelo.

## 5. Rasgos DSP usados en el proyecto

Un modelo de IA no recibe directamente la explicacion teorica de una modulacion. Necesita datos numericos. Por eso se extraen rasgos de cada senal.

Los rasgos se agrupan en cuatro familias principales:

### 5.1 Rasgos de amplitud

Estos rasgos describen como cambia la magnitud de la senal:

- Media de amplitud.
- Desviacion estandar de amplitud.
- Amplitud maxima.
- Relacion pico/promedio (PAPR).
- Percentiles de amplitud.
- Variacion de amplitud entre muestras.

Son utiles para modulaciones como ASK y QAM, donde la amplitud contiene informacion importante.

### 5.2 Rasgos IQ

Estos rasgos se calculan usando las componentes I y Q:

- Desviacion estandar de I.
- Desviacion estandar de Q.
- Media de valores absolutos de I y Q.
- Relacion cruzada entre I y Q.
- Variacion entre muestras consecutivas.

Ayudan a capturar la forma de la constelacion y la distribucion de energia entre los ejes.

### 5.3 Rasgos de fase

Los rasgos de fase son fundamentales para modulaciones PSK, como BPSK, QPSK y 8PSK. Incluyen:

- Variacion de fase.
- Promedio del cambio absoluto de fase.
- Relaciones de fase con retardos.
- Momentos circulares para distintos ordenes.

En los resultados del ablation study, los rasgos de fase fueron los mas fuertes de forma aislada. Esto es coherente porque varias clases del proyecto se distinguen principalmente por la fase.

### 5.4 Rasgos espectrales

Estos rasgos se extraen usando la FFT:

- Entropia espectral.
- Ancho de banda efectivo.
- Centroide espectral.
- Dispersion espectral.
- Pico espectral.
- Energia cerca del centro del espectro.

Son utiles para estudiar como se distribuye la energia en frecuencia. Por ejemplo, FSK tiende a mostrar componentes relacionadas con sus frecuencias de simbolo.

## 6. Modelo de inteligencia artificial

El modelo usado es una red neuronal multicapa sencilla, conocida como **MLP** (*Multilayer Perceptron*), implementada en NumPy. No se usa PyTorch ni TensorFlow en esta version, lo cual tiene dos ventajas:

1. El proyecto es ligero y facil de ejecutar.
2. Se entiende mejor que esta ocurriendo internamente.

El MLP recibe un vector de rasgos DSP y produce una probabilidad para cada modulacion:

```text
entrada: rasgos DSP
salida: BPSK, QPSK, 8PSK, 16QAM, 2FSK o 4ASK
```

El entrenamiento ajusta los pesos internos del modelo para que la salida correcta tenga mayor probabilidad. La funcion de perdida usada es una perdida de clasificacion multiclase, y la salida se interpreta usando softmax.

## 7. Baseline DSP

Para que el proyecto sea mas fuerte, no basta con decir que el MLP obtuvo buen accuracy. Es necesario compararlo con una referencia simple. Por eso se incluye un **baseline DSP por distancia a centroides**.

El baseline funciona asi:

1. Se extraen los mismos rasgos DSP.
2. Para cada modulacion, se calcula un centroide promedio en el espacio de rasgos.
3. Una nueva senal se clasifica segun el centroide mas cercano.

Este baseline representa una estrategia clasica y sencilla. Si la IA supera claramente este metodo, se puede justificar que el aprendizaje automatico aporta valor.

En los resultados actuales:

```text
Baseline DSP: 82.1%
MLP con rasgos DSP: 98.0%
```

Esto muestra una mejora absoluta aproximada de 15.9 puntos porcentuales.

## 8. Experimentos realizados

### 8.1 IA vs baseline DSP

Este experimento compara el MLP contra el baseline de distancia a centroides. Ambos usan los mismos rasgos DSP, por lo que la comparacion es justa: la diferencia principal esta en la capacidad del modelo para aprender fronteras de decision mas complejas.

**Pregunta que responde:**

```text
La IA realmente mejora sobre un metodo DSP clasico simple?
```

**Resultado:**

El MLP supera al baseline, lo cual indica que las relaciones entre rasgos no son completamente lineales o no se separan bien solo con distancia al promedio.

### 8.2 Robustez por SNR

Este experimento evalua el clasificador a diferentes niveles de ruido. Es importante porque en comunicaciones reales la calidad de la senal puede cambiar.

**Pregunta que responde:**

```text
A partir de que SNR el modelo es confiable?
```

**Lectura del resultado:**

El modelo tiene bajo rendimiento cuando el ruido es muy alto, especialmente en -5 dB y 0 dB. A partir de 5 dB mejora notablemente, y desde 10 dB alcanza resultados casi perfectos en el escenario simulado.

Esto es importante porque muestra el limite de operacion del sistema.

### 8.3 Ablation study de rasgos

El **ablation study** consiste en entrenar el modelo usando solo ciertos grupos de rasgos. La idea es observar que informacion aporta cada grupo.

Grupos evaluados:

- Amplitud.
- IQ.
- Fase.
- Espectro.
- Todos los rasgos.

**Pregunta que responde:**

```text
Que rasgos DSP son mas importantes para reconocer modulaciones?
```

**Lectura del resultado:**

Los rasgos de fase fueron los mas fuertes de manera aislada. Esto es coherente porque BPSK, QPSK y 8PSK se diferencian principalmente por estados de fase. Usar todos los rasgos produce el mejor resultado porque combina informacion de amplitud, fase, IQ y espectro.

### 8.4 Generalizacion a canales no ideales

En este experimento, el modelo se entrena con AWGN y luego se prueba en:

- AWGN.
- Rayleigh.
- Multipath.

**Pregunta que responde:**

```text
El modelo aprende la modulacion o depende demasiado del canal donde fue entrenado?
```

**Lectura del resultado:**

El modelo generaliza bien a AWGN y Rayleigh en la simulacion actual, pero baja en multipath. Esto indica que el multipath distorsiona la senal de una forma mas desafiante. Este resultado es valioso porque muestra una limitacion real y una oportunidad de mejora.

## 9. Matriz de confusion

La matriz de confusion muestra cuantas senales de cada clase fueron clasificadas correctamente o confundidas con otra clase. Es una herramienta mas informativa que el accuracy global porque permite ver donde falla el modelo.

Por ejemplo, si QPSK se confunde con 8PSK, esto no es sorprendente: ambas modulaciones se basan en fase y sus constelaciones pueden parecerse cuando hay ruido u offset.

La matriz de confusion ayuda a justificar que los errores no son aleatorios, sino relacionados con similitudes fisicas entre modulaciones.

## 10. Justificacion academica

Este proyecto esta justificado por varias razones:

### 10.1 Relevancia tecnica

La clasificacion automatica de modulaciones es una tarea relevante para receptores inteligentes, radio cognitiva y monitoreo de espectro. Un sistema capaz de identificar modulaciones puede adaptarse mejor a senales desconocidas.

### 10.2 Integracion de DSP e IA

El proyecto no reemplaza el DSP con IA. En cambio, usa DSP para extraer informacion significativa y usa IA para aprender patrones. Esto es importante porque mantiene cierta interpretabilidad.

### 10.3 Reproducibilidad

Como las senales se generan por codigo, cualquier persona puede repetir los experimentos, cambiar parametros y verificar resultados. Esto es una ventaja frente a depender de datos externos no controlados.

### 10.4 Bajo costo

No se necesita hardware especializado. El proyecto se puede desarrollar con Python y NumPy, lo cual lo hace accesible para un entorno universitario.

### 10.5 Valor pedagogico

El proyecto ayuda a comprender modulacion digital, constelaciones, SNR, ruido, FFT, canales y modelos de clasificacion. Por eso no solo tiene valor como implementacion, sino tambien como herramienta de aprendizaje.

### 10.6 Profundidad investigativa

La investigacion incluye comparaciones y pruebas:

- Comparacion contra baseline.
- Evaluacion por SNR.
- Ablation study.
- Generalizacion de canal.
- Diagnostico de cambio de dominio.
- Ruta progresiva de adaptacion: normalizacion objetivo, CORAL y fine-tuning.

Estas pruebas permiten defender el proyecto como investigacion y no solo como aplicacion.

## 11. Limitaciones actuales

Aunque los resultados iniciales son buenos, el proyecto tiene limitaciones:

- Las senales son sinteticas.
- El canal multipath usado es una aproximacion simple.
- No se han probado datasets reales como RadioML.
- El modelo no usa aprendizaje profundo sobre senales crudas.
- La adaptacion de dominio todavia esta en fase diagnostica.
- Aun no se implementan CORAL ni fine-tuning con muestras del dominio objetivo.
- No se mide tiempo de inferencia ni complejidad computacional.

Estas limitaciones no debilitan el proyecto; al contrario, permiten proponer mejoras claras para una siguiente etapa.

## 12. Posibles mejoras futuras

Para llevar el proyecto a un nivel mas cercano a publicacion o tesis, se recomienda:

1. Implementar normalizacion con estadisticas del dominio objetivo sin usar etiquetas.
2. Implementar CORAL para alinear la covarianza de los rasgos fuente y objetivo.
3. Probar fine-tuning con pocas muestras etiquetadas del dominio objetivo.
4. Repetir los experimentos de adaptacion con varias semillas y reportar media/desviacion estandar.
5. Evaluar con SNR mas fino, por ejemplo de -10 dB a 25 dB en pasos de 2 dB.
6. Comparar contra modelos adicionales como KNN, regresion logistica o SVM.
7. Probar datasets publicos como RadioML o capturas SDR reales.
8. Medir tiempo de inferencia, numero de parametros y costo computacional.
9. Analizar errores individuales mostrando constelacion, espectro y prediccion.

## 13. Explicacion sencilla para presentar al asesor

Una forma clara de explicar el proyecto es:

> Este proyecto parte del reconocimiento automatico de modulaciones digitales, pero no se queda solo en clasificar senales sinteticas. Primero se construye una base reproducible con senales IQ, rasgos DSP y un MLP ligero. Luego se mide que ocurre cuando el modelo entrenado en un dominio fuente AWGN se evalua en dominios objetivo mas dificiles, como bajo SNR y multipath. La parte investigativa esta en cuantificar esa caida y proponer una ruta de adaptacion de dominio para recuperar rendimiento sin depender inmediatamente de modelos profundos pesados.

En una reunion, se puede resumir asi:

- **Problema:** un clasificador entrenado en senales sinteticas puede perder rendimiento al aplicarse en condiciones mas realistas.
- **Metodo:** senales IQ + rasgos DSP + MLP ligero.
- **Comparacion:** MLP contra baseline DSP y contra escenarios sin adaptacion.
- **Robustez:** pruebas por SNR, canal, offset y multipath.
- **Explicabilidad:** ablation study de rasgos.
- **Resultado inicial:** el MLP funciona muy bien en fuente AWGN, pero cae hasta 60.1% en multipath severo.
- **Aporte:** proponer una ruta para reducir esa caida mediante adaptacion de dominio.
- **Diferenciacion:** no se vende como "primer AMC con IA"; se presenta como una investigacion ligera, interpretable y reproducible sobre recuperacion de rendimiento ante cambio de dominio.

## 14. Bibliografia y referencias recomendadas

- Dobre, O. A., Abdi, A., Bar-Ness, Y. y Su, W. (2007). *Survey of automatic modulation classification techniques: Classical approaches and new trends*. IET Communications. Disponible en: https://digitalcommons.njit.edu/fac_pubs/13459/
- O'Shea, T. J., Corgan, J. y Clancy, T. C. (2016). *Convolutional Radio Modulation Recognition Networks*. arXiv. Disponible en: https://arxiv.org/abs/1602.04105
- O'Shea, T. J., Corgan, J. y Clancy, T. C. (2016). *Unsupervised Representation Learning of Structured Radio Communication Signals*. arXiv. Disponible en: https://arxiv.org/abs/1604.07078
- O'Shea, T. y West, N. (2016). *RadioML 2016.10a Dataset*. Zenodo. Disponible en: https://zenodo.org/records/18397070
- Sun, B., Feng, J. y Saenko, K. (2016). *Return of Frustratingly Easy Domain Adaptation*. AAAI.
