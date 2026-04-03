import gradio as gr
from datasets import load_dataset
import pandas as pd
import random

# Load datasets from Hugging Face - JSONL format for chat
try:
    dataset = load_dataset("amkyawdev/AmkyawDev-Dataset")
    # Get labels from tags
    if "train" in dataset:
        train_data = dataset["train"]
        all_tags = set()
        for item in train_data:
            if "tags" in item:
                all_tags.add(item["tags"])
        label_names = list(all_tags) if all_tags else ["greeting", "coding", "culture", "food", "health", "math", "travel"]
    else:
        label_names = ["greeting", "coding", "culture", "food", "health", "math", "travel"]
except Exception as e:
    print(f"Error loading dataset: {e}")
    label_names = ["greeting", "coding", "culture", "food", "health", "math", "travel"]

LABELS = label_names

# Load chat data from JSONL - messages format
chat_pairs = []
chat_tags = []

try:
    from datasets import load_dataset
    chat_dataset = load_dataset("amkyawdev/AmkyawDev-Dataset", data_files="train.jsonl")
    chat_data = list(chat_dataset["train"])
    
    # Extract user/assistant pairs from messages format
    for item in chat_data:
        messages = item.get("messages", [])
        if len(messages) >= 3:
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if user_msg and assistant_msg:
                chat_pairs.append((user_msg.lower(), assistant_msg))
                chat_tags.append(item.get("tags", "other"))
    
    print(f"Loaded {len(chat_pairs)} chat pairs from dataset")
except Exception as e:
    print(f"Error loading chat data: {e}")
    chat_pairs = []
    chat_tags = []

# Fallback keyword responses
fallback_responses = {
    "သတင်း": "သတင်းသည်သင်တန်းစာသင်ပါး။ ပါသည်ကို ဖတ်ပါ။",
    "ကဗျာ": "မြန်မာကဗျာ ရေးလိုက်ပါ။ မန်းဝါးလောင်း အိပ်မက်မှာ ပါ။",
    "ဆေး": "ဆေးပပါး ဘာသာပါ။ ဆရာဝန်နဲ့ ပါသည်ကို သွားကြည့်ပါ။",
    "ဥပဒေ": "ဥပဒေစာတွေ ရှိပါ။ ဥပဒေအရ လုပ်ပါ။",
    "ပညာ": "ပါမိတ်ကား ပညာ သင်လိုက်ပါ။ ပါမိတ်ကား ပညာ ပါ။",
    "ဘာသာ": "ဘာသာစာသင်၊ ပါ။ သတင်း ပါ။",
    "စကား": "စကားပါတ်ခွဲ ပါ။ ဖြေးချက်ပီး ပါ။",
    "မိတ်ဆွေ": "မိတ်ဆွေများနဲ့ စကားပါတ်ပီး။ ပါတ်ပီး ပါ။",
}

# Chat function
def chat_response(user_input):
    """Chat response using dataset data"""
    if not user_input or len(user_input.strip()) == 0:
        return "ပါသည်ကို ရေးပါ။"
    
    user_lower = user_input.lower()
    
    # Try to find matching response from dataset
    for pattern, response in chat_pairs:
        if pattern in user_lower or user_lower in pattern:
            return response
    
    # Fallback to keyword matching
    for key, response in fallback_responses.items():
        if key in user_lower:
            return response
    
    # Default response
    return "နော်ကားဝီး သည် မြန်မာ NLP ပါတ်ပီး အက်ပလီကေးရှင်း ဖြစ်ပါတယ်။ ပါသည်ကို ပါးပါး ရှိပါ။"

# Load text generation data - using JSONL (same as chat)
text_gen_pairs = []
try:
    text_gen_dataset = load_dataset("amkyawdev/AmkyawDev-Dataset", data_files="train.jsonl")
    text_gen_data = list(text_gen_dataset["train"])
    # Extract prompt/completion pairs for text generation
    for item in text_gen_data:
        messages = item.get("messages", [])
        if len(messages) >= 3:
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if user_msg and assistant_msg:
                text_gen_pairs.append((user_msg, assistant_msg))
    print(f"Loaded {len(text_gen_pairs)} text gen pairs")
except Exception as e:
    print(f"Error loading text gen data: {e}")
    text_gen_pairs = []

# Simple keyword-based classifier (placeholder for ML model)
def simple_classify(text):
    """Simple keyword matching classifier"""
    text_lower = text.lower()
    
    keywords = {
        "greeting": ["နေကောင်း", "ဟောက်", "မင်္ဂလာ", "မှား"],
        "coding": ["python", "javascript", "html", "css", "java", "code", "ကုဒ်"],
        "culture": ["ရိုးရာ", "ပွဲ", "သင်္ကြန်", "တန်ဆောင်", "လက်ဝဲ"],
        "food": ["အစာ", "ဆန်", "မတ်", "မုန်", "ရှမ်း"],
        "health": ["ဆေး", "ကု", "ကျန်း", "စိတ်", "ချောင်း"],
        "economy": ["ငွေ", "တန်ဖိုး", "ဘဏ်", "လဲ"],
        "math": ["ပါ", "နှုန်း", "သင်္ချာ", "ပွဲ"],
        "travel": ["ခရီး", "ရန်ကုန်", "လည်း", "သွား", "ပါ"]
    }
    
    scores = {}
    for label, kws in keywords.items():
        score = sum(1 for kw in kws if kw in text_lower)
        scores[label] = score
    
    if max(scores.values()) > 0:
        predicted = max(scores, key=scores.get)
    else:
        predicted = "other"
    
    return predicted

# Chat function
def chat_response(user_input):
    """Simple keyword-based chat response"""
    if not user_input or len(user_input.strip()) == 0:
        return "ပါသည်ကို ရေးပါ။"
    
    user_lower = user_input.lower()
    
    # Keyword-based responses
    responses = {
        "သတင်း": "သတင်းသည်သင်တန်းစာသင်ပါး။ ပါသည်ကို ဖတ်ပါ။",
        "ကဗျာ": "မြန်မာကဗျာ ရေးလိုက်ပါ။ မန်းဝါးလောင်း အိပ်မက်မှာ ပါ။",
        "ဆေး": "ဆေးပပါး ဘာသာပါ။ ဆရာဝန်နဲ့ ပါသည်ကို သွားကြည့်ပါ။",
        "ဥပဒေ": "ဥပဒေစာတွေ ရှိပါ။ ဥပဒေအရ လုပ်ပါ။",
        "ပညာ": "ပါမိတ်ကား ပညာ သင်လိုက်ပါ။ ပါမိတ်ကား ပညာ ပါ။",
        "ဘာသာ": "ဘာသာစာသင်၊ ပါ။ သတင်း ပါ။",
        "စကား": "စကားပါတ်ခွဲ ပါ။ ဖြေးချက်ပီး ပါ။",
        "မိတ်ဆွေ": "မိတ်ဆွေများနဲ့ စကားပါတ်ပီး။ ပါတ်ပီး ပါ။",
    }
    
    for key, response in responses.items():
        if key in user_lower:
            return response
    
    # Default response
    return "နော်ကားဝီး သည် မြန်မာ NLP ပါတ်ပီး အက်ပလီကေးရှင်း ဖြစ်ပါတယ်။ ပါသည်ကို ပါးပါး ရှိပါ။"

# Text generation function
def generate_text(prompt):
    """Simple text generation based on prompt"""
    if not prompt or len(prompt.strip()) == 0:
        return "ပါသည်ကို ရေးပါ။"
    
    # Simple keyword matching
    if "သတင်း" in prompt:
        return "သတင်းသည်သင်တန်းစာသင်ပါး။ နိုင်ငံတော်သမိုင်းနဲ့ပါသည်ကို ရေးသားပါ။ ကမ္ဘာနှင့်အလယ်ပါတ်ပြင်းစာ ရှိပါ။"
    elif "ကဗျာ" in prompt:
        return "မြန်မာကဗျာ ရေးလိုက်ပါ။ မန်းဝါးလောင်း အိပ်မက်မှာ ပါ။ ပါတ်ပြားစွာ ရှိပါ။"
    elif "ဆေး" in prompt:
        return "ဆေးပပါး ဘာသာပါ။ ကျန်းမာရေး ပါတ်ပီး။ ဆရာဝန်နဲ့ ပါသည်ကို လမ်းညွှန်ပါ။"
    elif "ဥပဒေ" in prompt:
        return "ဥပဒေစာတွေ ရှိပါ။ စာခွင့်ပါတ်စာ ရှိပါ။ ဥပဒေအရ လုပ်ပါ။"
    elif "ပညာ" in prompt:
        return "ပါမိတ်ကား ပညာ သင်လိုက်ပါ။ သင်ယူခွင့်ပါ။ ကျောင်းသားများ ပါ။"
    elif "ဘာသာ" in prompt:
        return "ဘာသာစာသင်၊ ပါ။ သတင်း ပါ။ သံဃားစာပါ။"
    else:
        return "မြန်မာစာရေးသားပါ။ ပြင်ငန်း ရေးပါ။ ပါတ်ပီး ပါ။"


def predict(text):
    """Predict text category"""
    if not text or len(text.strip()) == 0:
        return {"error": "Please enter some text"}
    
    label = simple_classify(text)
    confidence = 0.85  # Placeholder confidence
    
    return {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "label": label,
        "confidence": confidence
    }

# Create Gradio interface with tabs
with gr.Blocks(title="AmkyawDev NLP") as demo:
    gr.Markdown("""
    <div align="center">
        <img src="https://raw.githubusercontent.com/amkyawdev/myanmar-lnp-dataset/main/logo.svg" width="120">
        <h1>🇲🇲 AmkyawDev NLP</h1>
        <p>Myanmar Language AI - Classification, Chat & Text Generation</p>
    </div>
    """)
    
    with gr.Tab("📊 Classification"):
        with gr.Row():
            with gr.Column(scale=1):
                input_text = gr.Textbox(
                    label="Enter Myanmar Text",
                    placeholder="မြန်မာစာရေးသားပါ... (e.g., သတင်းသည်သင်တန်းစာသင်ပါး)",
                    lines=5
                )
                submit_btn = gr.Button("Predict", variant="primary")
            
            with gr.Column(scale=1):
                output = gr.JSON(label="Prediction Result")
        
        submit_btn.click(predict, input_text, output)
        
        gr.Examples(
            examples=[
                ["သတင်းသည်သင်တန်းစာသင်ပါး"],
                ["မိတ်ဆွေများနဲ့ စကားပါတ်ပီး"],
                ["ကဗျာလုပ်သား ရေးပါ"],
                ["ဥပဒေစာတွေ ရှိပါ"],
                ["ဆေးပပါး ဘာသာပါ"],
            ],
            inputs=input_text,
        )
    
    with gr.Tab("💬 Chat"):
        gr.Markdown("### 💬 Myanmar Chat Bot")
        
        chatbot = gr.Chatbot(
            value=[],  # Initialize with empty list
            label="Chat History",
            height=400
        )
        
        msg = gr.Textbox(
            label="Your Message",
            placeholder="မြန်မာဘာသာဖြင့် ရေးပါ...",
            lines=2
        )
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("🗑️ Clear")
        
        def respond(message, history):
            if not message:
                return "", history
            response = chat_response(message)
            history = history or []
            history.append([message, response])
            return "", history
        
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
        
        gr.Examples(
            examples=[
                ["သတင်း ပါ။"],
                ["ကဗျာ ရေးပါ။"],
                ["ဆေးကုသပါ။"],
                ["ပညာ သင်ပါ။"],
            ],
            inputs=msg,
        )
    
    with gr.Tab("✍️ Text Generate"):
        gr.Markdown("### Myanmar Text Generation")
        
        with gr.Row():
            with gr.Column(scale=1):
                gen_prompt = gr.Textbox(
                    label="Prompt (အစ)",
                    placeholder="သတင်း... သို့မဟုတ် ကဗျာ...",
                    lines=3
                )
                gen_btn = gr.Button("Generate", variant="primary")
            
            with gr.Column(scale=1):
                gen_output = gr.Textbox(
                    label="Generated Text",
                    lines=5,
                    interactive=False
                )
        
        gen_btn.click(generate_text, gen_prompt, gen_output)
        gen_prompt.submit(generate_text, gen_prompt, gen_output)
        
        gr.Examples(
            examples=[
                ["သတင်း ဖတ်ပါ။"],
                ["ကဗျာ ရေးပါ။"],
                ["ဆေးကုသပါ။"],
                ["ဥပဒေ ပါ။"],
            ],
            inputs=gen_prompt,
        )
    
    gr.Markdown("""
    ---
    📚 **Dataset:** [AmkyawDev-Dataset](https://huggingface.co/datasets/amkyawdev/AmkyawDev-Dataset)  
    💻 **GitHub:** [amkyawdev/myanmar-lnp-dataset](https://github.com/amkyawdev/myanmar-lnp-dataset)
    """)

if __name__ == "__main__":
    demo.launch()