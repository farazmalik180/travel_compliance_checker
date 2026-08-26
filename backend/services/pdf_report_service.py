import io
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import green, red, black

def generate_compliance_report(profile: dict, final_result: dict) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = styles['Heading1']
    title_style.alignment = 1 # Center
    h2_style = styles['Heading2']
    normal_style = styles['Normal']
    
    status = final_result.get("status", "UNKNOWN")
    status_color = green if status == "CLEARED" else red
    status_style = ParagraphStyle(
        'StatusStyle',
        parent=styles['Heading2'],
        textColor=status_color
    )
    
    story = []
    
    # Add Logo if exists
    logo_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'pim_logo.jpg')
    if os.path.exists(logo_path):
        try:
            img = Image(logo_path, width=100, height=100)
            story.append(img)
            story.append(Spacer(1, 0.2 * 72))
        except Exception:
            pass
    
    # Title
    story.append(Paragraph("PIM Passenger Compliance Report", title_style))
    story.append(Spacer(1, 0.5 * 72))
    
    # Passenger Profile
    story.append(Paragraph("Passenger Profile Summary", h2_style))
    story.append(Paragraph(f"<b>Destination:</b> {profile.get('destination', 'N/A')}", normal_style))
    story.append(Paragraph(f"<b>Visa Category:</b> {profile.get('visa_category', 'N/A')}", normal_style))
    story.append(Paragraph(f"<b>Passport History:</b> {profile.get('passport_history', 'N/A')}", normal_style))
    if "bank_funds" in profile:
        story.append(Paragraph(f"<b>Bank Funds:</b> {profile.get('bank_funds')}", normal_style))
    story.append(Spacer(1, 0.3 * 72))
    
    # Final Decision
    story.append(Paragraph("Final Agent Decision", h2_style))
    story.append(Paragraph(f"<b>Status:</b> {status}", status_style))
    story.append(Paragraph(f"<b>Compliance Score:</b> {final_result.get('compliance_score', 'N/A')}/100", normal_style))
    story.append(Spacer(1, 0.3 * 72))
    
    # Verified Items
    story.append(Paragraph("Verified Requirements", h2_style))
    for item in final_result.get("verified_items", []):
        story.append(Paragraph(f"+ {item}", normal_style)) 
    story.append(Spacer(1, 0.2 * 72))
    
    # Missing Items
    if final_result.get("missing_or_incomplete_requirements"):
        story.append(Paragraph("Missing / Incomplete Requirements", h2_style))
        for item in final_result.get("missing_or_incomplete_requirements", []):
            story.append(Paragraph(f"- <font color='red'>{item}</font>", normal_style)) 
        story.append(Spacer(1, 0.2 * 72))
        
    # FIA Reference
    story.append(Paragraph("FIA Policy Reference", h2_style))
    story.append(Paragraph(final_result.get('fia_rule_reference', 'No reference available'), normal_style))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
