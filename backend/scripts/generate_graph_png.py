import urllib.request
import base64
import os

mermaid_graph = """graph TD
    __start__((START))
    __end__((END))
    
    extract[Extract Document Info]
    retrieve[Retrieve Rules RAG]
    enhanced_scrutiny[Enhanced Scrutiny Check]
    verify[Verify Compliance]
    audit[Audit Feedback]
    
    __start__ --> extract
    extract --> retrieve
    
    retrieve -- "passport_history == 'Fresh'" --> enhanced_scrutiny
    retrieve -- "passport_history == 'Experienced'" --> verify
    
    enhanced_scrutiny --> verify
    verify --> audit
    audit --> __end__
    
    %% Styling
    classDef default fill:#1E293B,stroke:#94A3B8,stroke-width:2px,color:#F8FAFC;
    classDef startend fill:#0F172A,stroke:#10B981,stroke-width:3px,color:#F8FAFC;
    classDef conditional fill:#334155,stroke:#F59E0B,stroke-width:2px,color:#F8FAFC,stroke-dasharray: 5 5;
    
    class __start__,__end__ startend;
    class enhanced_scrutiny conditional;
"""

def generate_png():
    # Base64 encode the mermaid string
    graph_bytes = mermaid_graph.encode('utf-8')
    base64_bytes = base64.b64encode(graph_bytes)
    base64_string = base64_bytes.decode('utf-8')
    
    # URL for Mermaid Ink
    url = f"https://mermaid.ink/img/{base64_string}"
    
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "compliance_workflow.png")
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"Successfully generated graph visualization at {output_path}")
    except Exception as e:
        print(f"Failed to generate graph PNG: {e}")

if __name__ == "__main__":
    generate_png()
