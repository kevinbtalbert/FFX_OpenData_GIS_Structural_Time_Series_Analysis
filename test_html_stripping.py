#!/usr/bin/env python3
"""
Test script to validate HTML stripping from chat messages
"""

from html.parser import HTMLParser
import html as html_module

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

# Test cases - simulating what Azure OpenAI might return
test_cases = [
    # Case 1: Plain text (ideal)
    {
        "input": "Based on the current annual growth rate of 1.72%, the predicted property value for the next 6 months is approximately $368.5 billion.",
        "expected": "Based on the current annual growth rate of 1.72%, the predicted property value for the next 6 months is approximately $368.5 billion."
    },
    # Case 2: HTML wrapped response
    {
        "input": "<div class=\"chat-message assistant-message\"><strong>🤖 AI Assistant</strong><br><span>Based on the current annual growth rate of 1.72%, the predicted property value for the next 6 months is approximately $368.5 billion.</span></div>",
        "expected": "🤖 AI AssistantBased on the current annual growth rate of 1.72%, the predicted property value for the next 6 months is approximately $368.5 billion."
    },
    # Case 3: Paragraph tags
    {
        "input": "<p>If property values decline, Fairfax County's revenue risk is tied to its reliance on real estate taxes.</p>",
        "expected": "If property values decline, Fairfax County's revenue risk is tied to its reliance on real estate taxes."
    },
    # Case 4: Multiple nested tags
    {
        "input": "<div><strong>Key Point:</strong> <span>The forecast shows <em>positive growth</em> of 1.5%.</span></div>",
        "expected": "Key Point: The forecast shows positive growth of 1.5%."
    }
]

print("Testing HTML Stripping Logic")
print("=" * 60)

all_passed = True
for i, test in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print(f"Input: {test['input'][:100]}...")
    
    # Strip HTML
    result = strip_html_tags(test['input'])
    
    # Escape for display
    escaped = html_module.escape(result)
    
    print(f"After stripping: {result[:100]}...")
    print(f"After escaping: {escaped[:100]}...")
    print(f"Expected: {test['expected'][:100]}...")
    
    # Check if it matches (allowing for whitespace differences)
    if result.strip() == test['expected'].strip():
        print("✓ PASSED")
    else:
        print("✗ FAILED")
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✓ All tests passed!")
else:
    print("✗ Some tests failed")

# Now test the actual chat display HTML generation
print("\n" + "=" * 60)
print("Testing Chat Display HTML Generation")
print("=" * 60)

# Simulate a chat message
test_message = {
    'role': 'assistant',
    'content': '<p>Based on the current annual growth rate of 1.72%, the predicted property value is $368.5B.</p>'
}

print(f"\nOriginal message content: {test_message['content']}")

# Process it like we do in the app
clean_content = strip_html_tags(test_message['content'])
print(f"After stripping HTML: {clean_content}")

escaped_content = html_module.escape(clean_content)
print(f"After escaping: {escaped_content}")

# Build the HTML like we do in the app
chat_html = f"""
<div class="chat-message assistant-message">
    <strong>🤖 AI Assistant</strong><br>
    <span>{escaped_content}</span>
</div>
"""

print(f"\nFinal HTML to render:")
print(chat_html)

print("\n" + "=" * 60)
print("Expected behavior:")
print("- The <p> tags should be stripped")
print("- The text content should remain")
print("- Special chars like < > & should be escaped")
print("- Our wrapper HTML should NOT be escaped")
