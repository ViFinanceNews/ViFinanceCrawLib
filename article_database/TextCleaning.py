import textacy.preprocessing as tp
import regex as re
class TextCleaning():
    def __init__(self):
        return
    
    def clean_text(self, text_str, punctuation=True):
        text = tp.normalize.whitespace(text_str)  # Normalize spaces
        if punctuation:
            text = tp.remove.punctuation(text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
if __name__ == "__main__":
    tc = TextCleaning()
    test_str = "   Hello, world!!!    This   is a   test...   "
    clean_text = tc.clean_text(test_str, punctuation=False)
    print(clean_text)