# Resumen para asesor

## Mensaje central

El proyecto ya no debe presentarse como un simple clasificador de modulaciones. La vision mas fuerte es: **adaptacion de dominio para reconocer modulaciones cuando el modelo pasa de entrenamiento sintetico controlado a condiciones objetivo mas realistas**.

## Estado actual

- Pipeline funcional: senales IQ, canal simulado, 36 rasgos DSP y MLP ligero.
- Accuracy base: 97.9%.
- MLP de soporte: 98.0%.
- Baseline DSP: 82.1%.
- Fase 1 de cambio de dominio implementada.

## Resultado clave

| Escenario | Accuracy objetivo | Caida |
|---|---:|---:|
| source_holdout_awgn | 100.0% | 0.0 pp |
| target_awgn_low_snr | 82.9% | 17.1 pp |
| target_rayleigh | 95.7% | 4.3 pp |
| target_multipath | 73.3% | 26.7 pp |
| target_harsh_multipath | 60.1% | 39.9 pp |

## Que pedirle al asesor

1. Confirmar que el enfoque final sea adaptacion de dominio.
2. Aprobar Fase 2: normalizacion objetivo sin etiquetas.
3. Decidir si la version final debe usar solo simulacion o tambien RadioML/SDR.
