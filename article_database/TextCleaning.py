import textacy.preprocessing as tp
import regex as re
class TextCleaning():
    def __init__(self):
        return
    
    def clean_text(self, text_str, punctuation=True):
        """
        Clean and normalize a text string by handling whitespace and optional punctuation removal.

        Args:
            text_str (str): The input text to be cleaned.
            punctuation (bool, optional): If True, remove all punctuation from the text. 
                                        Defaults to True.

        Returns:
            str: A cleaned version of the input text with normalized spacing 
                and optionally no punctuation.
        """
        text = tp.normalize.whitespace(text_str)  # Normalize spaces
        if punctuation:
            text = tp.remove.punctuation(text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
