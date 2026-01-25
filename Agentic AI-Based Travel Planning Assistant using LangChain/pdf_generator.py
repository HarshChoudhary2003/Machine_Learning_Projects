from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

def generate_pdf(trip_data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 40
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, trip_data["Trip Summary"])

    c.setFont("Helvetica", 10)
    y -= 30

    for trip in trip_data["Trips"]:
        c.drawString(40, y, f"Destination: {trip['Destination']}")
        y -= 15

        c.drawString(60, y, f"Flight: {trip['Flight Selected']['airline']} ₹{trip['Flight Selected']['price']}")
        y -= 15

        c.drawString(60, y, f"Hotel: {trip['Hotel Selected']['name']} ({trip['Hotel Selected']['stars']}⭐)")
        y -= 15

        c.drawString(60, y, "Itinerary:")
        y -= 15

        for day, places in trip["Day-wise Itinerary"].items():
            c.drawString(80, y, f"{day}: {', '.join(places)}")
            y -= 15

        c.drawString(60, y, f"Total Budget: ₹{trip['Budget Breakdown']['Total']}")
        y -= 30

        if y < 100:
            c.showPage()
            y = height - 40

    c.save()
    buffer.seek(0)
    return buffer
