import pytest
from api.preprocess import (
    clean_text,
    normalize_unicode,
    remove_whitespace,
    remove_special_chars,
    is_myanmar_char,
    tokenize_by_word,
    MyanmarPreprocessor,
)


class TestPreprocessing:
    """Tests for Myanmar preprocessing."""
    
    def test_is_myanmar_char(self):
        """Test Myanmar character detection."""
        # Valid Myanmar characters
        assert is_myanmar_char("က") == True
        assert is_myanmar_char("မ") == True
        
        # Not Myanmar
        assert is_myanmar_char("a") == False
        assert is_myanmar_char("1") == False
    
    def test_normalize_unicode(self):
        """Test Unicode normalization."""
        text = "သတင်း"
        normalized = normalize_unicode(text, "NFC")
        
        assert normalized == text
    
    def test_remove_whitespace(self):
        """Test whitespace removal."""
        text = "သတင်း   သည်"
        cleaned = remove_whitespace(text)
        
        assert cleaned == "သတင်း သည်"
    
    def test_remove_special_chars(self):
        """Test special character removal."""
        text = "သတင်း! @#"
        
        # Keep Myanmar
        cleaned = remove_special_chars(text, keep_myanmar=True)
        assert "သတင်း" in cleaned
        
        # Remove Myanmar too
        cleaned = remove_special_chars(text, keep_myanmar=False)
        assert "!" in cleaned
    
    def test_clean_text(self):
        """Test full text cleaning."""
        text = "  သတင်း   သည်  "
        cleaned = clean_text(text)
        
        assert cleaned.startswith("သတင်း")
        assert cleaned.endswith("သည်")
    
    def test_tokenize_by_word(self):
        """Test word tokenization."""
        text = "သတင်း သည် သင်တန်း"
        tokens = tokenize_by_word(text)
        
        assert len(tokens) == 3
        assert tokens[0] == "သတင်း"
    
    def test_preprocessor(self):
        """Test MyanmarPreprocessor class."""
        preprocessor = MyanmarPreprocessor()
        
        text = "  သတင်း   သည်  "
        cleaned = preprocessor(text)
        
        assert "သတင်း" in cleaned
        assert "  " not in cleaned
    
    def test_preprocessor_transform(self):
        """Test batch processing."""
        preprocessor = MyanmarPreprocessor()
        
        texts = ["သတင်း သည်", "လူမှုကွပ်"]
        processed = preprocessor.transform(texts)
        
        assert len(processed) == 2
        assert all(isinstance(t, str) for t in processed)