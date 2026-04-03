"""Data augmentation module for Myanmar NLP.

Provides synonym replacement and back-translation augmentation.
"""

import random
from typing import Optional

# Basic Myanmar synonym dictionary
# In production, use comprehensive Myanmar WordNet or dictionary
MYANMAR_SYNONYMS = {
    "သတင်း": ["သတင်းအရောက်", "သတင်းသည်", "သတင်းပါး"],
    "ကောက်ပါ": ["ချိတ်ဆက်", "နွယ်ယက်", "ထောက်ပါ"],
    "နည်းပါး": ["သောက်သည်", "လမ်းစဉ်", "နည်းလမ်း"],
    "ပါတ်ခွဲ": ["ပါဒါဟဒါ", "ခွဲပါဟခွဲ", "ပါတ်ပြား"],
    "သည်": ["ဖြစ်ပပါး", "ရှိပါး", "ဖြစ်ပီး"],
}


class MyanmarAugmenter:
    """Data augmenter for Myanmar text.
    
    Provides various augmentation strategies.
    """
    
    def __init__(self,
                synonym_prob: float = 0.15,
                random_insert_prob: float = 0.1,
                random_swap_prob: float = 0.1,
                synonyms_dict: Optional[dict] = None):
        """Initialize augmenter.
        
        Args:
            synonym_prob: Probability of synonym replacement
            random_insert_prob: Probability of random insertion
            random_swap_prob: Probability of random swap
            synonyms_dict: Custom synonyms dictionary
        """
        self.synonym_prob = synonym_prob
        self.random_insert_prob = random_insert_prob
        self.random_swap_prob = random_swap_prob
        self.synonyms_dict = synonyms_dict or MYANMAR_SYNONYMS
    
    def synonym_replace(self, text: str) -> str:
        """Replace words with synonyms.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        words = text.split()
        augmented_words = []
        
        for word in words:
            if word in self.synonyms_dict and random.random() < self.synonym_prob:
                synonyms = self.synonyms_dict[word]
                word = random.choice(synonyms)
            augmented_words.append(word)
        
        return " ".join(augmented_words)
    
    def random_insert(self, text: str) -> str:
        """Randomly insert words.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        if random.random() < self.random_insert_prob:
            # Common filler words in Myanmar
            fillers = ["လည်းပါး", "ပါး", "ပြီး", "ဟုတ်ပါး"]
            insert_word = random.choice(fillers)
            
            words = text.split()
            if words:
                pos = random.randint(0, len(words))
                words.insert(pos, insert_word)
                
                return " ".join(words)
        
        return text
    
    def random_swap(self, text: str) -> str:
        """Randomly swap adjacent words.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        if random.random() < self.random_swap_prob:
            words = text.split()
            
            if len(words) >= 2:
                pos = random.randint(0, len(words) - 2)
                words[pos], words[pos + 1] = words[pos + 1], words[pos]
                
                return " ".join(words)
        
        return text
    
    def augment(self, text: str) -> str:
        """Apply random augmentation strategies.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        text = self.synonym_replace(text)
        text = self.random_insert(text)
        text = self.random_swap(text)
        
        return text
    
    def __call__(self, text: str) -> str:
        """Apply augmentation.
        
        Args:
            text: Input text
            
        Returns:
            Augmented text
        """
        return self.augment(text)
    
    def augment_batch(self, texts: list[str]) -> list[str]:
        """Augment batch of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of augmented texts
        """
        return [self.augment(text) for text in texts]


def back_translate_example():
    """Placeholder for back-translation augmentation.
    
    In production, integrate with translation APIs
    like Google Translate, DeepL, or OpenAI.
    
    Example:
        >>> # Using OpenAI API for back-translation
        >>> import openai
        >>> 
        >>> def back_translate(text, target_lang="en"):
        ...     # Translate to English and back to Myanmar
        ...     response = openai.ChatCompletion.create(
        ...         model="gpt-4",
        ...         messages=[
        ...             {"role": "system", "content": "You are a translator."},
        ...             {"role": "user", "content": f"Translate to {target_lang}: {text}"}
        ...         ]
        ...     )
        ...     return response.choices[0].message.content
    """
    pass