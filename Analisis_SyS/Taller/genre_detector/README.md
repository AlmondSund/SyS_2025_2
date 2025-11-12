# Detector de género musical (FFT + k-NN)

Este paquete concentra toda la lógica usada en el taller para el punto **1.1.e**.

## Componentes principales

- `config.SpectralConfig`: hiper-parámetros de audio y STFT.
- `knn.KNNSpectralClassifier`: clasificador k-NN sobre la magnitud promedio del espectro.
- `pipeline`: funciones de alto nivel (`auto_fit_and_save`, `classify_with_model`) y CLI (`python -m Analisis_SyS.Taller.genre_detector.pipeline`).
- `service`: integra los artefactos locales (`dataset.csv`, `artifacts/`, `audio_cache/`) y expone `ensure_default_model` y `predict_from_link`.

## Uso rápido

```bash
# Entrenar (descarga automáticamente las pistas del dataset CSV)
python -m Analisis_SyS.Taller.genre_detector.pipeline \
    --dataset Analisis_SyS/Taller/dataset.csv \
    --path-col link \
    --label-col genre \
    --out-model Analisis_SyS/Taller/artifacts/genre_knn.joblib \
    --cache-dir Analisis_SyS/Taller/audio_cache

# Clasificar un enlace de YouTube una vez el modelo exista
python -m Analisis_SyS.Taller.genre_detector.pipeline \
    --out-model Analisis_SyS/Taller/artifacts/genre_knn.joblib \
    --classify \"https://www.youtube.com/watch?v=dQw4w9WgXcQ\"
```

En los notebooks basta con llamar a `predict_from_link` para obtener el género estimado, los vecinos más cercanos y las rutas locales asociadas.
