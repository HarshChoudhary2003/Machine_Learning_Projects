from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def build_battery_report(summary: dict) -> bytes:
    """
    summary = {
        "title": str,
        "battery_percent": float,
        "predicted_minutes": float,
        "efficiency_score": int,
        "efficiency_label": str,
        "recommendations": list[str],
        "context": str,
    }
    """
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, y, summary.get("title", "Battery Optimization Report"))
    y -= 30

    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Generated at: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    y -= 20

    c.drawString(50, y, f"Context: {summary.get('context', 'N/A')}")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Battery Overview")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Battery percent: {summary.get('battery_percent', 'N/A')}%")
    y -= 15

    minutes = summary.get("predicted_minutes", 0)
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    c.drawString(60, y, f"Predicted remaining time: {hours} hours {mins} minutes")
    y -= 25

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Energy Efficiency")
    y -= 18

    c.setFont("Helvetica", 11)
    c.drawString(
        60,
        y,
        f"Efficiency score: {summary.get('efficiency_score', 'N/A')}/100 "
        f"({summary.get('efficiency_label', 'N/A')})",
    )
    y -= 25

    recs = summary.get("recommendations", [])
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Recommended Actions")
    y -= 18

    c.setFont("Helvetica", 11)
    if not recs:
        c.drawString(60, y, "No urgent actions needed.")
        y -= 15
    else:
        for r in recs:
            if y < 60:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
            c.drawString(60, y, f"- {r}")
            y -= 15

    c.save()
    buffer.seek(0)

    # Return PDF bytes
    return buffer.getvalue()
