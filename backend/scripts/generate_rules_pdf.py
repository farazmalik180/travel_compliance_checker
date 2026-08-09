from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def create_pdf(filename):
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Heading1']
    h2_style = styles['Heading2']
    body_style = styles['Normal']
    
    story = []
    
    story.append(Paragraph("FIA Departure Guidelines and Exit Rules", title_style))
    story.append(Spacer(1, 0.2 * inch))
    
    # General Rules
    story.append(Paragraph("1. Work Visas", h2_style))
    story.append(Paragraph(
        "Any passenger traveling on a Work Visa must present a valid passport, a valid visa, a Protector Stamp issued by the Bureau of Emigration, and a valid Work Permit (where applicable for the destination country).", 
        body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("2. Tourist / Visit Visas", h2_style))
    story.append(Paragraph(
        "Passengers traveling on a Tourist or Visit Visa must present a valid passport, a confirmed return ticket, a confirmed hotel booking with proof of advance payment, and sufficient financial proof. Sufficient funds are defined as a minimum equivalent of 2,000 EUR or a valid credit card with adequate limits. The traveler must also demonstrate a sound travel profile indicating intent to return.", 
        body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("3. Government Servants", h2_style))
    story.append(Paragraph(
        "Any Government Servant traveling abroad must provide an original No Objection Certificate (NOC) explicitly issued by their parent government department or ministry.", 
        body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    # Specific Rules
    story.append(Paragraph("4. High-Scrutiny Watchlist Destinations", h2_style))
    story.append(Paragraph(
        "Travel to any of the 15 monitored countries requires heightened risk scrutiny and rigorous profile validation. These countries include: Saudi Arabia, Iran, Iraq, Türkiye, Qatar, Azerbaijan, Kuwait, Kyrgyzstan, Russia, Egypt, Libya, Ethiopia, Senegal, Mauritania, and Kenya.", 
        body_style))
    story.append(Spacer(1, 0.1 * inch))
    
    story.append(Paragraph("5. Cambodia Specific Rules", h2_style))
    story.append(Paragraph(
        "For travel to Cambodia (which is also treated under high-scrutiny), authorities must explicitly verify a confirmed return ticket on the exact same PNR as the outbound ticket, a confirmed hotel booking, at least 1,000 USD in show money (cash), and verified visa documentation.", 
        body_style))
    
    doc.build(story)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(current_dir), "data")
    pdf_path = os.path.join(data_dir, "fia_exit_rules.pdf")
    create_pdf(pdf_path)
