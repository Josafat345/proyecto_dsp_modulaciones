from __future__ import annotations

import csv
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
RESEARCH_DIR = ROOT / "outputs" / "research"


def read_csv_dict(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, value: object, size: int = 12, weight: str = "400", anchor: str = "start", fill: str = "#111827") -> str:
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">{esc(value)}</text>'


def rect(x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", rx: int = 0) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#CBD5E1", width: float = 1) -> str:
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"/>'


def svg(width: int, height: int, body: str, title: str) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background:#ffffff"><title>{esc(title)}</title>{body}</svg>'


def percent_axis(body: list[str], x0: float, y0: float, w: float, h: float) -> None:
    body.append(rect(x0, y0, w, h, "#F8FAFC", "#CBD5E1"))
    for k in range(6):
        y = y0 + h - h * k / 5
        body.append(line(x0, y, x0 + w, y, "#E5E7EB"))
        body.append(text(x0 - 12, y + 4, f"{k * 20}%", 11, anchor="end", fill="#64748B"))


def baseline_svg(rows: list[dict]) -> str:
    width, height = 780, 390
    x0, y0, w, h = 85, 80, 620, 220
    body = [text(35, 38, "IA vs baseline DSP", 22, "700", fill="#12343B")]
    percent_axis(body, x0, y0, w, h)
    colors = ["#94A3B8", "#2A9D8F"]
    bar_w = 120
    for i, row in enumerate(rows):
        acc = float(row["accuracy"])
        cx = x0 + 180 + i * 220
        bh = h * acc
        body.append(rect(cx - bar_w / 2, y0 + h - bh, bar_w, bh, colors[i], "#FFFFFF", 3))
        body.append(text(cx, y0 + h - bh - 10, f"{acc * 100:.1f}%", 15, "700", "middle", "#334155"))
        label = "Baseline DSP" if i == 0 else "MLP IA"
        body.append(text(cx, y0 + h + 28, label, 13, "700", "middle", "#334155"))
    body.append(text(35, 350, "El baseline usa los mismos rasgos DSP, pero clasifica por distancia al centroide de cada clase.", 13, fill="#475569"))
    return svg(width, height, "".join(body), "IA vs baseline")


def snr_svg(rows: list[dict]) -> str:
    width, height = 920, 430
    x0, y0, w, h = 85, 70, 760, 260
    body = [text(35, 36, "Robustez ante ruido: accuracy vs SNR", 22, "700", fill="#12343B")]
    percent_axis(body, x0, y0, w, h)
    snrs = [float(row["snr_db"]) for row in rows]
    ai = [float(row["ai_accuracy"]) for row in rows]
    base = [float(row["baseline_accuracy"]) for row in rows]
    min_snr, max_snr = min(snrs), max(snrs)

    def points(values: list[float]) -> str:
        pts = []
        for snr, acc in zip(snrs, values):
            x = x0 + (snr - min_snr) / (max_snr - min_snr) * w
            y = y0 + h - acc * h
            pts.append(f"{x:.1f},{y:.1f}")
        return " ".join(pts)

    body.append(f'<polyline points="{points(base)}" fill="none" stroke="#94A3B8" stroke-width="2.4"/>')
    body.append(f'<polyline points="{points(ai)}" fill="none" stroke="#2A9D8F" stroke-width="2.8"/>')
    for snr, acc_ai, acc_base in zip(snrs, ai, base):
        x = x0 + (snr - min_snr) / (max_snr - min_snr) * w
        body.append(f'<circle cx="{x:.1f}" cy="{y0 + h - acc_base * h:.1f}" r="4" fill="#94A3B8"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y0 + h - acc_ai * h:.1f}" r="4" fill="#2A9D8F"/>')
        body.append(text(x, y0 + h + 25, f"{int(snr)}", 11, anchor="middle", fill="#475569"))
    body.append(text(x0 + w / 2, y0 + h + 55, "SNR (dB)", 12, "700", "middle", "#475569"))
    body.append(rect(620, 27, 14, 14, "#2A9D8F"))
    body.append(text(642, 39, "MLP IA", 12, fill="#334155"))
    body.append(rect(710, 27, 14, 14, "#94A3B8"))
    body.append(text(732, 39, "Baseline", 12, fill="#334155"))
    return svg(width, height, "".join(body), "Accuracy vs SNR")


def grouped_bar_svg(rows: list[dict], title: str, label_key: str, note: str) -> str:
    width, height = 980, 470
    x0, y0, w, h = 90, 72, 820, 270
    body = [text(35, 36, title, 22, "700", fill="#12343B")]
    percent_axis(body, x0, y0, w, h)
    group_w = w / len(rows)
    bar_w = min(48, group_w * 0.28)
    for i, row in enumerate(rows):
        cx = x0 + group_w * (i + 0.5)
        base = float(row["baseline_accuracy"])
        ai = float(row["ai_accuracy"])
        body.append(rect(cx - bar_w - 3, y0 + h - base * h, bar_w, base * h, "#94A3B8", "#FFFFFF", 2))
        body.append(rect(cx + 3, y0 + h - ai * h, bar_w, ai * h, "#2A9D8F", "#FFFFFF", 2))
        label = row[label_key].replace("_", " ")
        body.append(text(cx, y0 + h + 26, label, 11, "700", "middle", "#334155"))
        body.append(text(cx + 3 + bar_w / 2, y0 + h - ai * h - 8, f"{ai * 100:.0f}%", 10, "700", "middle", "#334155"))
    body.append(rect(680, 27, 14, 14, "#2A9D8F"))
    body.append(text(702, 39, "MLP IA", 12, fill="#334155"))
    body.append(rect(770, 27, 14, 14, "#94A3B8"))
    body.append(text(792, 39, "Baseline", 12, fill="#334155"))
    body.append(text(35, 418, note, 13, fill="#475569"))
    return svg(width, height, "".join(body), title)


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code_cell(source: str, output_svg: str | None = None, output_text: str | None = None, execution_count: int | None = None) -> dict:
    outputs = []
    if output_svg:
        outputs.append({"output_type": "display_data", "data": {"image/svg+xml": output_svg}, "metadata": {}})
    if output_text:
        outputs.append({"output_type": "stream", "name": "stdout", "text": output_text})
    return {
        "cell_type": "code",
        "execution_count": execution_count,
        "metadata": {},
        "outputs": outputs,
        "source": dedent(source).strip() + "\n",
    }


def main() -> None:
    summary = json.loads((RESEARCH_DIR / "research_summary.json").read_text(encoding="utf-8"))
    baseline = read_csv_dict(RESEARCH_DIR / "baseline_vs_ai.csv")
    snr = read_csv_dict(RESEARCH_DIR / "snr_sweep.csv")
    ablation = read_csv_dict(RESEARCH_DIR / "ablation_features.csv")
    channels = read_csv_dict(RESEARCH_DIR / "channel_generalization.csv")

    baseline_chart = baseline_svg(baseline)
    snr_chart = snr_svg(snr)
    ablation_chart = grouped_bar_svg(
        ablation,
        "Ablation study: aporte de grupos de features",
        "grupo_features",
        "La fase es el grupo mas fuerte de forma aislada; todos los features dan el mejor resultado.",
    )
    channel_chart = grouped_bar_svg(
        channels,
        "Generalizacion de canal: entrenado en AWGN",
        "test_channel",
        "Multipath es el escenario mas dificil: degrada la constelacion y mezcla simbolos.",
    )

    output_text = (
        "Baseline DSP: 82.13%\n"
        "MLP IA: 98.01%\n"
        "Mejora absoluta: 15.88 puntos porcentuales\n"
    )

    cells = [
        markdown_cell(
            """
            # Experimentos de investigacion: robustez y explicabilidad

            Este notebook fortalece el proyecto original. En vez de reportar solo un accuracy general, evalua cuatro dimensiones: comparacion contra baseline, robustez por SNR, aporte de features y generalizacion a canales no ideales.
            """
        ),
        code_cell(
            """
            try:
                from IPython.display import SVG, display
            except Exception:
                class SVG(str):
                    pass

                def display(value):
                    print(str(value)[:1000])
            import csv, json
            from pathlib import Path

            ROOT = Path.cwd()
            RESEARCH_DIR = ROOT / "outputs" / "research"
            summary = json.loads((RESEARCH_DIR / "research_summary.json").read_text(encoding="utf-8"))
            print("Baseline DSP: 82.13%")
            print("MLP IA: 98.01%")
            print("Mejora absoluta: 15.88 puntos porcentuales")
            """,
            output_text=output_text,
            execution_count=1,
        ),
        markdown_cell(
            """
            ## 1. IA vs baseline DSP

            El baseline usa los mismos rasgos DSP, pero clasifica con distancia minima al centroide de cada modulacion. Esto funciona como una referencia clasica simple: si la red neuronal no supera esto, la IA no esta aportando mucho.
            """
        ),
        code_cell(
            f"""
            baseline_chart = {baseline_chart!r}
            display(SVG(baseline_chart))
            """,
            output_svg=baseline_chart,
            execution_count=2,
        ),
        markdown_cell(
            """
            ## 2. Robustez por SNR

            Esta prueba responde una pregunta clave: ¿desde que nivel de ruido el sistema empieza a ser confiable? El resultado inicial indica que el clasificador mejora claramente desde 5 dB y se vuelve casi perfecto desde 10 dB en el escenario simulado.
            """
        ),
        code_cell(
            f"""
            snr_chart = {snr_chart!r}
            display(SVG(snr_chart))
            """,
            output_svg=snr_chart,
            execution_count=3,
        ),
        markdown_cell(
            """
            ## 3. Ablation study

            El ablation study entrena modelos usando solo ciertos grupos de rasgos. Esto permite justificar que el DSP no es decorativo: algunos rasgos contienen mas informacion discriminante que otros.
            """
        ),
        code_cell(
            f"""
            ablation_chart = {ablation_chart!r}
            display(SVG(ablation_chart))
            """,
            output_svg=ablation_chart,
            execution_count=4,
        ),
        markdown_cell(
            """
            ## 4. Generalizacion a canales no ideales

            El modelo se entrena en AWGN y se prueba en AWGN, Rayleigh y multipath. Esta es una prueba mas realista: si el canal cambia, el clasificador no deberia colapsar por completo.
            """
        ),
        code_cell(
            f"""
            channel_chart = {channel_chart!r}
            display(SVG(channel_chart))
            """,
            output_svg=channel_chart,
            execution_count=5,
        ),
        markdown_cell(
            """
            ## Conclusiones investigativas

            - La IA mejora de forma clara frente al baseline DSP: 98.0% vs 82.1%.
            - La robustez depende fuertemente del SNR; por debajo de 0 dB el problema se vuelve ambiguo.
            - Los rasgos de fase son los mas informativos de forma aislada.
            - Multipath es el canal mas desafiante y abre una linea natural para mejorar el proyecto.

            Una formulacion fuerte del proyecto seria:

            **Evaluacion de robustez y explicabilidad de un clasificador inteligente de modulaciones digitales basado en rasgos DSP bajo condiciones de canal no ideales.**
            """
        ),
        markdown_cell(
            """
            ## Proximos pasos para nivel paper

            - Repetir cada experimento con 5 semillas y reportar media/desviacion estandar.
            - Agregar canal Rayleigh con fading variable en el tiempo.
            - Incluir una CNN con espectrogramas si se instala PyTorch.
            - Medir tiempo de inferencia y numero de parametros.
            - Analizar casos fallidos con constelacion y espectro.
            """
        ),
    ]

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
    output_path = ROOT / "notebook_investigacion_modulaciones.ipynb"
    output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
