"""Preprocessing module for Myanmar text.

Handles Unicode normalization, cleaning, and tokenization for Myanmar text.
"""

import re
import unicodedata
from typing import Optional

# Myanmar Unicode ranges
MYANMAR_RANGE = (0x1000, 0x109F)
MYANMAR_EXTENDED_A_RANGE = (0xAA60, 0xAA7F)
MYANMAR_EXTENDED_B_RANGE = (0xA9E0, 0xA9FF)


def is_myanmar_char(char: str) -> bool:
    """Check if character is Myanmar script.
    
    Args:
        char: Single character to check
        
    Returns:
        True if character is Myanmar
    """
    if len(char) != 1:
        raise ValueError("Expected single character")
    
    code = ord(char)
    return (MYANMAR_RANGE[0] <= code <= MYANMAR_RANGE[1] or
            MYANMAR_EXTENDED_A_RANGE[0] <= code <= MYANMAR_EXTENDED_A_RANGE[1] or
            MYANMAR_EXTENDED_B_RANGE[0] <= code <= MYANMAR_EXTENDED_B_RANGE[1])


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text.
    
    Args:
        text: Input text
        form: Unicode normalization form (NFC, NFD, NFKC, NFKD)
        
    Returns:
        Normalized text
    """
    return unicodedata.normalize(form, text)


def remove_whitespace(text: str) -> str:
    """Remove extra whitespace from text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with single space
    text = re.sub(r"\s+", " ", text)
    # Remove leading/trailing whitespace
    return text.strip()


def remove_special_chars(text: str, keep_myanmar: bool = True) -> str:
    """Remove special characters from text.
    
    Args:
        text: Input text
        keep_myanmar: Whether to keep Myanmar characters
        
    Returns:
        Cleaned text
    """
    if keep_myanmar:
        # Keep Myanmar characters, English, numbers, and basic punctuation
        pattern = r"[^\w\s.,၊။။.!()?\-:;\"\']"
    else:
        # Only keep word characters and basic punctuation
        pattern = r"[^\w\s.,.!()?\-:;\"\']"
    
    return re.sub(pattern, "", text)


def normalize_whitespace_variants(text: str) -> str:
    """Normalize whitespace variants in Myanmar text.
    
    Converts various whitespace characters to regular space.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Replace Unicode whitespace variants
    whitespace_pattern = r"[\u00A0\u1680\u180E\u2000-\u200A\u202F\u205F\u3000]"
    return re.sub(whitespace_pattern, " ", text)


def normalize_zawgyi(text: str) -> str:
    """Normalize Zawgyi to Unicode.
    
    Note: This is a basic implementation. For production,
    consider using proper Zawgyi detection libraries.
    
    Args:
        text: Input text (may contain Zawgyi)
        
    Returns:
        Text with Zawgyi converted to Unicode
    """
    # Basic Zawgyi to Unicode mappings for common characters
    zawgyi_map = {
        "\u1039": "\u1000\u103B",  # Ka-hto -> Ka + medial
        "\u104E": "\u101E\u103D",  # Luns
        # Add more mappings as needed
    }
    
    for zawgyi, unicode_char in zawgyi_map.items():
        text = text.replace(zawgyi, unicode_char)
    
    return text


def tokenize_by_char(text: str) -> list[str]:
    """Tokenize Myanmar text by character.
    
    Args:
        text: Input text
        
    Returns:
        List of characters
    """
    return list(text)


def tokenize_by_word(text: str, delimiter: Optional[str] = None) -> list[str]:
    """Tokenize Myanmar text by word.
    
    Args:
        text: Input text
        delimiter: Custom delimiter (default: whitespace)
        
    Returns:
        List of words
    """
    if delimiter:
        return text.split(delimiter)
    return text.split()


def tokenize_syllable(text: str) -> list[str]:
    """Tokenize Myanmar text by syllable.
    
    This is a basic syllable tokenizer. For more accurate
    segmentation, consider using morphological analysis.
    
    Args:
        text: Input text
        
    Returns:
        List of syllables
    """
    # Simple syllable segmentation based on whitespace
    # and Myanmar character boundaries
    syllables = []
    current = []
    
    for char in text:
        if char.isspace():
            if current:
                syllables.append("".join(current))
                current = []
        else:
            current.append(char)
    
    if current:
        syllables.append("".join(current))
    
    return syllables


def clean_text(text: str, 
              normalize_unicode_form: str = "NFC",
              remove_extra_whitespace: bool = True,
              remove_special: bool = False,
              normalize_zawgyi_to_unicode: bool = True) -> str:
    """Clean and normalize Myanmar text.
    
    Args:
        text: Input text
        normalize_unicode_form: Unicode normalization form
        remove_extra_whitespace: Remove extra whitespace
        remove_special: Remove special characters
        normalize_zawgyi_to_unicode: Convert Zawgyi to Unicode
        
    Returns:
        Cleaned text
    """
    if not text:
        return text
    
    # Normalize whitespace variants
    text = normalize_whitespace_variants(text)
    
    # Normalize Unicode
    text = normalize_unicode(text, normalize_unicode_form)
    
    # Normalize Zawgyi
    if normalize_zawgyi_to_unicode:
        text = normalize_zawgyi(text)
    
    # Remove special characters
    if remove_special:
        text = remove_special_chars(text)
    
    # Remove extra whitespace
    if remove_extra_whitespace:
        text = remove_whitespace(text)
    
    return text


class MyanmarPreprocessor:
    """Myanmar text preprocessor.
    
    Provides configurable preprocessing pipeline for Myanmar text.
    """
    
    def __init__(self,
                normalize_unicode: bool = True,
                unicode_form: str = "NFC",
                clean_whitespace: bool = True,
                remove_special: bool = False,
                normalize_zawgyi: bool = True,
                lowercase: bool = False):
        """Initialize preprocessor.
        
        Args:
            normalize_unicode: Apply Unicode normalization
            unicode_form: Unicode normalization form
            clean_whitespace: Remove extra whitespace
            remove_special: Remove special characters
            normalize_zawgyi: Convert Zawgyi to Unicode
            lowercase: Convert to lowercase (rarely used for Myanmar)
        """
        self.normalize_unicode = normalize_unicode
        self.unicode_form = unicode_form
        self.clean_whitespace = clean_whitespace
        self.remove_special = remove_special
        self.normalize_zawgyi = normalize_zawgyi
        self.lowercase = lowercase
    
    def __call__(self, text: str) -> str:
        """Preprocess text.
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        return clean_text(
            text,
            normalize_unicode_form=self.unicode_form,
            remove_extra_whitespace=self.clean_whitespace,
            remove_special=self.remove_special,
            normalize_zawgyi_to_unicode=self.normalize_zawgyi
        )
    
    def transform(self, texts: list[str]) -> list[str]:
        """Preprocess batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of preprocessed texts
        """
        return [self(text) for text in texts]