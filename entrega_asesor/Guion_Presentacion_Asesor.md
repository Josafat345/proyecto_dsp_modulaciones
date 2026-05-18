# Guion breve para presentar al asesor

## 1. Abrir con el problema

"Profesor, el proyecto partio como clasificacion automatica de modulaciones, pero encontramos que eso ya esta bastante trabajado. Por eso lo estoy reorientando hacia adaptacion de dominio: que pasa cuando entreno con senales sinteticas controladas y pruebo en condiciones mas realistas."

## 2. Mostrar lo que ya funciona

- Generador de senales IQ.
- Rasgos DSP interpretables.
- Modelo MLP ligero.
- Baseline DSP.
- Experimentos por SNR, ablation y canal.

## 3. Mostrar el hallazgo importante

"En AWGN controlado el modelo llega a 100%, pero en multipath severo baja a 60.1%. Esa caida es la oportunidad de investigacion."

## 4. Defender la novedad realista

"No quiero venderlo como el primer clasificador de modulaciones con IA. La vision es evaluar recuperacion de rendimiento con tecnicas ligeras e interpretables de adaptacion de dominio."

## 5. Pedir decision

- Validar el enfoque.
- Aprobar Fase 2.
- Decidir si luego incorporamos RadioML o SDR.
