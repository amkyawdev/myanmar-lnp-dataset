import pytest
from api.data_loader import LNPDataLoader, load_data, save_data


class TestLNPDataLoader:
    """Tests for LNPDataLoader."""
    
    @pytest.fixture
    def temp_jsonl(self, tmp_path):
        """Create temporary JSONL file."""
        filepath = tmp_path / "test.jsonl"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write('{"text": "သတင်း", "label": 0}\n')
            f.write('{"text": "လူမှုကွပ်", "label": 1}\n')
        
        return str(filepath)
    
    def test_load_jsonl(self, temp_jsonl):
        """Test loading JSONL file."""
        loader = LNPDataLoader()
        df = loader.load_file(temp_jsonl)
        
        assert len(df) == 2
        assert "text" in df.columns
        assert "label" in df.columns
    
    def test_iter_file(self, temp_jsonl):
        """Test iterating JSONL file."""
        loader = LNPDataLoader()
        
        records = list(loader.iter_file(temp_jsonl))
        
        assert len(records) == 2
        assert records[0]["text"] == "သတင်း"
    
    def test_save_jsonl(self, tmp_path):
        """Test saving JSONL file."""
        data = [
            {"text": "သတင်း", "label": 0},
            {"text": "လူမှုကွပ်", "label": 1},
        ]
        
        output_path = tmp_path / "output.jsonl"
        
        loader = LNPDataLoader()
        loader.save_jsonl(data, str(output_path))
        
        assert output_path.exists()
    
    def test_save_csv(self, tmp_path):
        """Test saving CSV file."""
        import pandas as pd
        
        df = pd.DataFrame({
            "text": ["သတင်း", "လူမှုကွပ်"],
            "label": [0, 1]
        })
        
        output_path = tmp_path / "output.csv"
        
        loader = LNPDataLoader()
        loader.save_csv(df, str(output_path))
        
        assert output_path.exists()