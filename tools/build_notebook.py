from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from signal_generator import MODULATIONS, apply_channel, generate_clean_signal  # noqa: E402


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_frame(width: int, height: int, body: str, title: str = "") -> str:
    title_tag = f"<title>{esc(title)}</title>" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" style="background:#ffffff">{title_tag}{body}</svg>'
    )


def text(x: float, y: float, content: str, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111827") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(content)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#CBD5E1", width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def polyline(points: list[tuple[float, float]], stroke: str, width: float = 1.8) -> str:
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"/>'


def map_series(values: np.ndarray, x0: float, y0: float, w: float, h: float, y_min: float, y_max: float) -> list[tuple[float, float]]:
    if len(values) <= 1:
        return []
    span = y_max - y_min if abs(y_max - y_min) > 1e-12 else 1.0
    points = []
    for i, value in enumerate(values):
        x = x0 + w * i / (len(values) - 1)
        y = y0 + h - h * (float(value) - y_min) / span
        points.append((x, y))
    return points


def iq_time_svg(signal: np.ndarray) -> str:
    width, height = 920, 360
    x0, y0, w, h = 70, 70, 800, 220
    n = 260
    i = np.real(signal[:n])
    q = np.imag(signal[:n])
    y_min = float(min(i.min(), q.min()))
    y_max = float(max(i.max(), q.max()))
    body = []
    body.append(text(40, 32, "Señal IQ recibida en el tiempo (ejemplo QPSK con ruido)", 20, "700", fill="#12343B"))
    body.append(rect(x0, y0, w, h, "#F8FAFC", "#CBD5E1"))
    for k in range(6):
        yy = y0 + h * k / 5
        body.append(line(x0, yy, x0 + w, yy, "#E5E7EB"))
    body.append(polyline(map_series(i, x0, y0, w, h, y_min, y_max), "#156082", 2))
    body.append(polyline(map_series(q, x0, y0, w, h, y_min, y_max), "#E76F51", 2))
    body.append(text(x0, y0 + h + 34, "muestras", 12, fill="#475569"))
    body.append(text(14, y0 + 18, "amplitud", 12, fill="#475569"))
    body.append(rect(720, 28, 14, 14, "#156082"))
    body.append(text(742, 40, "I", 13, fill="#334155"))
    body.append(rect(770, 28, 14, 14, "#E76F51"))
    body.append(text(792, 40, "Q", 13, fill="#334155"))
    return svg_frame(width, height, "".join(body), "Señal IQ")


def constellation_svg(rng: np.random.Generator) -> str:
    width, height = 1040, 620
    panel_w, panel_h = 300, 230
    margin_x, margin_y = 45, 70
    gap_x, gap_y = 35, 55
    body = [text(40, 34, "Constelaciones ideales por modulación", 22, "700", fill="#12343B")]

    for idx, modulation in enumerate(MODULATIONS):
        row = idx // 3
        col = idx % 3
        x0 = margin_x + col * (panel_w + gap_x)
        y0 = margin_y + row * (panel_h + gap_y)
        clean = generate_clean_signal(modulation, n_symbols=170, samples_per_symbol=8, rng=rng)
        points_complex = clean[::8]
        xs = np.real(points_complex)
        ys = np.imag(points_complex)
        lim = max(1.25, float(np.max(np.abs(np.r_[xs, ys]))) * 1.15)
        body.append(rect(x0, y0, panel_w, panel_h, "#F8FAFC", "#CBD5E1"))
        body.append(line(x0 + panel_w / 2, y0 + 12, x0 + panel_w / 2, y0 + panel_h - 28, "#CBD5E1"))
        body.append(line(x0 + 20, y0 + panel_h / 2, x0 + panel_w - 20, y0 + panel_h / 2, "#CBD5E1"))
        for x_val, y_val in zip(xs, ys):
            px = x0 + 20 + (x_val + lim) / (2 * lim) * (panel_w - 40)
            py = y0 + panel_h - 28 - (y_val + lim) / (2 * lim) * (panel_h - 40)
            body.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="3.2" fill="#2A9D8F" fill-opacity="0.65"/>')
        body.append(text(x0 + panel_w / 2, y0 - 12, modulation, 15, "700", "middle", "#334155"))
        body.append(text(x0 + panel_w - 24, y0 + panel_h / 2 - 8, "I", 11, fill="#64748B"))
        body.append(text(x0 + panel_w / 2 + 8, y0 + 24, "Q", 11, fill="#64748B"))
    return svg_frame(width, height, "".join(body), "Constelaciones")


def spectrum_svg(rng: np.random.Generator) -> str:
    width, height = 980, 440
    x0, y0, w, h = 80, 75, 840, 270
    examples = ["BPSK", "16QAM", "2FSK"]
    colors = {"BPSK": "#156082", "16QAM": "#2A9D8F", "2FSK": "#E76F51"}
    body = [text(40, 36, "Espectro de potencia comparativo", 22, "700", fill="#12343B")]
    body.append(rect(x0, y0, w, h, "#F8FAFC", "#CBD5E1"))
    for k in range(6):
        yy = y0 + h * k / 5
        body.append(line(x0, yy, x0 + w, yy, "#E5E7EB"))
    min_db, max_db = -65.0, 0.0
    for modulation in examples:
        clean = generate_clean_signal(modulation, n_symbols=128, samples_per_symbol=8, rng=rng)
        noisy = apply_channel(clean, snr_db=13.0, rng=rng)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(noisy * np.hanning(len(noisy))))) ** 2
        spectrum_db = 10 * np.log10(spectrum / np.max(spectrum) + 1e-12)
        spectrum_db = np.clip(spectrum_db, min_db, max_db)
        sampled = spectrum_db[::4]
        body.append(polyline(map_series(sampled, x0, y0, w, h, min_db, max_db), colors[modulation], 1.9))
    body.append(text(x0, y0 + h + 36, "frecuencia normalizada", 12, fill="#475569"))
    body.append(text(20, y0 + 24, "dB", 12, fill="#475569"))
    lx = x0 + 570
    for j, modulation in enumerate(examples):
        body.append(rect(lx + j * 95, 32, 14, 14, colors[modulation]))
        body.append(text(lx + 20 + j * 95, 44, modulation, 12, fill="#334155"))
    return svg_frame(width, height, "".join(body), "Espectro")


def confusion_svg(matrix: np.ndarray, labels: list[str]) -> str:
    width, height = 760, 680
    x0, y0, cell = 160, 95, 74
    max_val = max(1, int(matrix.max()))
    body = [text(40, 36, "Matriz de confusión del clasificador", 22, "700", fill="#12343B")]
    body.append(text(x0 + cell * 3, 68, "Predicho", 14, "700", "middle", "#334155"))
    body.append(text(54, y0 + cell * 3, "Real", 14, "700", "middle", "#334155"))
    for i, label in enumerate(labels):
        body.append(text(x0 + i * cell + cell / 2, y0 - 18, label, 12, "700", "middle", "#334155"))
        body.append(text(x0 - 18, y0 + i * cell + cell / 2 + 5, label, 12, "700", "end", "#334155"))
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = int(matrix[r, c])
            alpha = value / max_val
            red = int(238 - 190 * alpha)
            green = int(250 - 90 * alpha)
            blue = int(252 - 80 * alpha)
            fill = f"rgb({red},{green},{blue})"
            body.append(rect(x0 + c * cell, y0 + r * cell, cell, cell, fill, "#FFFFFF"))
            color = "#FFFFFF" if alpha > 0.62 else "#111827"
            body.append(text(x0 + c * cell + cell / 2, y0 + r * cell + cell / 2 + 5, str(value), 16, "700", "middle", color))
    body.append(text(40, height - 42, "La mayor confusión aparece entre QPSK y 8PSK, porque ambas codifican información principalmente en fase.", 13, fill="#475569"))
    return svg_frame(width, height, "".join(body), "Matriz de confusion")


def accuracy_bar_svg(summary: dict) -> str:
    labels = list(summary["per_class"].keys())
    values = [summary["per_class"][label]["accuracy"] for label in labels]
    width, height = 900, 420
    x0, y0, w, h = 80, 65, 760, 260
    body = [text(40, 36, "Exactitud por modulación", 22, "700", fill="#12343B")]
    body.append(rect(x0, y0, w, h, "#F8FAFC", "#CBD5E1"))
    for k in range(6):
        y = y0 + h - h * k / 5
        body.append(line(x0, y, x0 + w, y, "#E5E7EB"))
        body.append(text(x0 - 12, y + 4, f"{k * 20}%", 11, anchor="end", fill="#64748B"))
    bar_w = w / len(labels) * 0.58
    for i, (label, value) in enumerate(zip(labels, values)):
        cx = x0 + (i + 0.5) * w / len(labels)
        bh = h * value
        body.append(rect(cx - bar_w / 2, y0 + h - bh, bar_w, bh, "#2A9D8F"))
        body.append(text(cx, y0 + h - bh - 8, f"{value * 100:.1f}%", 12, "700", "middle", "#334155"))
        body.append(text(cx, y0 + h + 24, label, 12, "700", "middle", "#334155"))
    return svg_frame(width, height, "".join(body), "Accuracy por clase")


def feature_concept_svg() -> str:
    width, height = 920, 360
    stages = [
        ("Señal IQ", "I/Q complejo\ncon ruido y offset"),
        ("DSP", "FFT, amplitud,\nfase y energía"),
        ("Features", "24+ rasgos\ncompactos"),
        ("IA", "MLP ligero\nen NumPy"),
        ("Salida", "BPSK, QPSK,\n8PSK, QAM..."),
    ]
    body = [text(40, 36, "Flujo metodológico del proyecto", 22, "700", fill="#12343B")]
    for i, (title, detail) in enumerate(stages):
        x = 45 + i * 175
        y = 95
        body.append(rect(x, y, 140, 125, "#EAF4F4", "#CBD5E1", 8))
        body.append(text(x + 70, y + 34, title, 15, "700", "middle", "#12343B"))
        for j, line_text in enumerate(detail.split("\n")):
            body.append(text(x + 70, y + 68 + j * 20, line_text, 12, "400", "middle", "#334155"))
        if i < len(stages) - 1:
            body.append(line(x + 145, y + 62, x + 168, y + 62, "#64748B", 2))
            body.append(f'<polygon points="{x+168:.1f},{y+62:.1f} {x+158:.1f},{y+56:.1f} {x+158:.1f},{y+68:.1f}" fill="#64748B"/>')
    body.append(text(45, 285, "La ventaja investigativa es que cada etapa puede medirse, modificarse y justificarse: no es una caja negra completa.", 14, fill="#475569"))
    return svg_frame(width, height, "".join(body), "Flujo metodologico")


def read_confusion(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    labels = rows[0][1:]
    matrix = np.asarray([[int(v) for v in row[1:]] for row in rows[1:]], dtype=int)
    return labels, matrix


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code_cell(source: str, output_svg: str | None = None, output_text: str | None = None, execution_count: int | None = None) -> dict:
    outputs = []
    if output_svg is not None:
        outputs.append({"output_type": "display_data", "data": {"image/svg+xml": output_svg}, "metadata": {}})
    if output_text is not None:
        outputs.append({"output_type": "stream", "name": "stdout", "text": output_text})
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": dedent(source).strip() + "\n",
    }


def main() -> None:
    summary_path = ROOT / "outputs" / "summary.json"
    confusion_path = ROOT / "outputs" / "confusion_matrix.csv"
    if not summary_path.exists() or not confusion_path.exists():
        raise FileNotFoundError("Ejecuta src/experiment.py antes de construir el notebook.")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    labels, matrix = read_confusion(confusion_path)
    rng = np.random.default_rng(123)
    qpsk = generate_clean_signal("QPSK", n_symbols=128, samples_per_symbol=8, rng=rng)
    qpsk_noisy = apply_channel(qpsk, snr_db=12.0, rng=rng)
    assets_dir = ROOT / "outputs" / "notebook_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    svgs = {
        "flujo_metodologico.svg": feature_concept_svg(),
        "constelaciones.svg": constellation_svg(np.random.default_rng(123)),
        "iq_tiempo.svg": iq_time_svg(qpsk_noisy),
        "espectro.svg": spectrum_svg(np.random.default_rng(123)),
        "accuracy_por_clase.svg": accuracy_bar_svg(summary),
        "matriz_confusion.svg": confusion_svg(matrix, labels),
    }
    for name, svg in svgs.items():
        (assets_dir / name).write_text(svg, encoding="utf-8")

    metrics_text = (
        f"Accuracy entrenamiento: {summary['train_accuracy'] * 100:.2f}%\n"
        f"Accuracy prueba: {summary['test_accuracy'] * 100:.2f}%\n"
        f"Clases: {', '.join(summary['clases'])}\n"
    )

    cells = [
        markdown_cell(
            """
            # Reconocimiento inteligente de modulaciones digitales con DSP e IA

            Este notebook resume y justifica el proyecto de investigación. La idea es generar señales IQ sintéticas, aplicar efectos de canal, extraer rasgos de procesamiento digital de señales y entrenar un modelo de IA ligero para clasificar modulaciones digitales.
            """
        ),
        markdown_cell(
            """
            ## Pregunta de investigación

            ¿Puede un modelo de IA ligero reconocer modulaciones digitales usando rasgos DSP compactos, incluso cuando las señales tienen ruido, fase aleatoria, offset de frecuencia y variación de ganancia?

            **Hipótesis:** sí. Los rasgos de amplitud, fase, espectro y constelación conservan suficiente información física para separar modulaciones como BPSK, QPSK, 8PSK, 16QAM, 2FSK y 4ASK.
            """
        ),
        code_cell(
            """
            from pathlib import Path
            import csv, json, sys
            import numpy as np
            try:
                from IPython.display import SVG, display
            except Exception:
                class SVG(str):
                    pass

                def display(value):
                    print(str(value)[:1000])

            ROOT = Path.cwd()
            if (ROOT / "src").exists():
                sys.path.insert(0, str(ROOT / "src"))
            else:
                # Si se abre desde otra carpeta, ajustar ROOT al directorio del proyecto.
                ROOT = Path("proyecto_dsp_modulaciones").resolve()
                sys.path.insert(0, str(ROOT / "src"))

            summary = json.loads((ROOT / "outputs" / "summary.json").read_text(encoding="utf-8"))
            print(f"Accuracy entrenamiento: {summary['train_accuracy']*100:.2f}%")
            print(f"Accuracy prueba: {summary['test_accuracy']*100:.2f}%")
            print("Clases:", ", ".join(summary["clases"]))
            """,
            output_text=metrics_text,
            execution_count=1,
        ),
        code_cell(
            """
            # Flujo conceptual del sistema: de señal IQ a predicción de modulación.
            display(SVG(flujo_metodologico_svg))
            """,
            output_svg=svgs["flujo_metodologico.svg"],
            execution_count=2,
        ),
        markdown_cell(
            """
            ## Justificación DSP

            Las modulaciones digitales no solo se distinguen por el tiempo. También cambian la forma de la constelación, la distribución de fase, el comportamiento espectral y la dinámica de amplitud. Por eso el proyecto extrae rasgos como:

            - estadísticos de amplitud I/Q;
            - derivadas de fase;
            - entropía espectral;
            - ancho de banda efectivo;
            - relaciones de fase con retardos;
            - relación pico/promedio.
            """
        ),
        code_cell(
            """
            # Las constelaciones muestran por qué el problema es físicamente separable.
            display(SVG(constelaciones_svg))
            """,
            output_svg=svgs["constelaciones.svg"],
            execution_count=3,
        ),
        code_cell(
            """
            # Ejemplo de señal recibida: componentes I y Q en el dominio del tiempo.
            display(SVG(iq_tiempo_svg))
            """,
            output_svg=svgs["iq_tiempo.svg"],
            execution_count=4,
        ),
        code_cell(
            """
            # Comparación espectral: FSK tiende a mostrar concentración en frecuencias separadas.
            display(SVG(espectro_svg))
            """,
            output_svg=svgs["espectro.svg"],
            execution_count=5,
        ),
        markdown_cell(
            """
            ## Justificación del modelo de IA

            El modelo usado es una red neuronal MLP implementada en NumPy. Esto hace que el proyecto sea ligero y explicable: no depende de un framework grande, pero sí demuestra aprendizaje supervisado sobre rasgos DSP.

            El flujo de entrenamiento es:

            1. generar señales sintéticas por clase;
            2. aplicar canal con ruido y offset;
            3. extraer rasgos DSP;
            4. normalizar rasgos;
            5. entrenar MLP;
            6. evaluar con matriz de confusión.
            """
        ),
        code_cell(
            """
            # Accuracy por clase a partir de outputs/summary.json.
            display(SVG(accuracy_por_clase_svg))
            """,
            output_svg=svgs["accuracy_por_clase.svg"],
            execution_count=6,
        ),
        code_cell(
            """
            # Matriz de confusión a partir de outputs/confusion_matrix.csv.
            display(SVG(matriz_confusion_svg))
            """,
            output_svg=svgs["matriz_confusion.svg"],
            execution_count=7,
        ),
        markdown_cell(
            f"""
            ## Lectura de resultados

            El experimento inicial obtuvo **{summary['test_accuracy'] * 100:.2f}% de accuracy en prueba**. Las clases BPSK, 8PSK, 2FSK y 4ASK quedaron perfectamente separadas en esta corrida. La mayor confusión aparece entre QPSK y 8PSK, algo razonable porque ambas son modulaciones por fase y pueden parecerse cuando hay ruido u offset.

            Esto justifica el proyecto porque combina:

            - señales generadas por código;
            - fenómenos reales de canal;
            - rasgos DSP interpretables;
            - un modelo de IA entrenable;
            - evaluación cuantitativa.
            """
        ),
        markdown_cell(
            """
            ## Próximas mejoras recomendadas

            - Evaluar accuracy por niveles de SNR: 0 dB, 5 dB, 10 dB, 15 dB, 20 dB.
            - Agregar canal Rayleigh o multipath.
            - Comparar contra un clasificador basado en reglas.
            - Convertir señales a espectrogramas y probar una CNN.
            - Crear una interfaz que muestre señal, FFT, constelación y predicción.
            """
        ),
    ]

    # Variables SVG para que el notebook siga siendo ejecutable y editable.
    cells.insert(
        3,
        code_cell(
            """
            # SVGs pre-generados para que el notebook sea liviano y no dependa de Matplotlib.
            ASSET_DIR = ROOT / "outputs" / "notebook_assets"

            def load_svg(name):
                return (ASSET_DIR / name).read_text(encoding="utf-8")

            flujo_metodologico_svg = load_svg("flujo_metodologico.svg")
            constelaciones_svg = load_svg("constelaciones.svg")
            iq_tiempo_svg = load_svg("iq_tiempo.svg")
            espectro_svg = load_svg("espectro.svg")
            accuracy_por_clase_svg = load_svg("accuracy_por_clase.svg")
            matriz_confusion_svg = load_svg("matriz_confusion.svg")
            """,
            execution_count=2,
        ),
    )

    # Renumber executed cells after insertion.
    count = 1
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = count
            count += 1

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    output_path = ROOT / "notebook_resumen_modulaciones.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
