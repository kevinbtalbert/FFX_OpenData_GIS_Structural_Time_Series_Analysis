#!/usr/bin/env python3
"""
Test the actual fix - simulate what happens in the app
"""

from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = []
    def handle_data(self, d):
        self.text.append(d)
    def get_data(self):
        return ''.join(self.text)

def strip_html_tags(html_text):
    """Completely remove all HTML tags and return only text content"""
    s = MLStripper()
    s.feed(html_text)
    return s.get_data()

# Simulate what Azure OpenAI is returning (based on the screenshot)
azure_response = """<div class="chat-message assistant-message">
<strong> AI Assistant</strong><br>
<span>Assuming a consistent 1.47% annual growth rate, the 6-month predicted property value is approximately $370.2 billion.</span>
</div>"""

print("Simulating the app's chat display logic:")
print("=" * 60)
print(f"\n1. Azure OpenAI returns:\n{azure_response}\n")

# Strip HTML (this is what we do when receiving the response)
clean_content = strip_html_tags(azure_response)
print(f"2. After stripping HTML:\n{clean_content}\n")

# Build our chat HTML (WITHOUT escaping)
chat_html = f"""<div class="chat-message assistant-message">
    <strong> AI Assistant</strong><br>
    <span>{clean_content}</span>
</div>"""

print(f"3. Final HTML to render with st.markdown(..., unsafe_allow_html=True):")
print(chat_html)

print("\n" + "=" * 60)
print("Expected result in browser:")
print("A gray box with ' AI Assistant' and the clean text below it")
print("\nActual content that should display:")
print(clean_content)
