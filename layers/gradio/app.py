import gradio as gr
from gradio.components import ChatMessage
from datasets import load_dataset
import pandas as pd
import random


# Load datasets from Hugging Face - JSONL format for chat
try:
    dataset = load_dataset("amkyawdev/AmkyawDev-Dataset")
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


# Load chat data
chat_pairs = []
chat_tags = []


try:
    chat_dataset = load_dataset("amkyawdev/AmkyawDev-Dataset", data_files="train.jsonl")
    chat_data = list(chat_dataset["train"])
    for item in chat_data:
        messages = item.get("messages", [])
        if len(messages) >= 3:
            user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            if user_msg and assistant_msg:
                chat_pairs.append((user_msg, assistant_msg))
                chat_tags.append(item.get("tags", "other"))
    print(f"Loaded {len(chat_pairs)} chat pairs from dataset")
except Exception as e:
    print(f"Error loading chat data: {e}")
    chat_pairs = []
    chat_tags = []


# Fallback responses (with corrected Myanmar words)
fallback_responses = {
    "သတင်း": "သတင်းသည်သင်တန်းစာသင်ပါး။ နိုင်ငံတော်သမိုင်းနဲ့ရေးသားပါ။ ကမ္ဘာနှင့် အလယ်ပိုင်းတွင် အရေးကြီးသည့် အချက်များ ပါဝင်ပါ။",
    "ကဗျာ": "မြန်မာကဗျာ ရေးလိုက်ပါ။ အိပ်မက်မှာ ပါ။ အကျယ်ပြန့်စွာ ရှိပါ။",
    "ဆေး": "ဆေးပပါး ဘာသာပါ။ ဆရာဝန်နဲ့ သွားကြည့်ပါ။",
    "ဥပဒေ": "ဥပဒေစာတွေ ရှိပါ။ စာခွင့် အရ လုပ်ပါ။",
    "ပညာ": "ပညာ သင်လိုက်ပါ။ သင်ယူခွင့် ပါ။ ကျောင်းသားများ ပါ။",
    "ဘာသာ": "ဘာသာစာသင်၊ ပါ။ သတင်း ပါ။ သံဃားစာပါ။",
    "စကား": "စကား ခွဲပါ။ ဖြေချက် ပါ။",
    "မိတ်ဆွေ": "မိတ်ဆွေများနဲ့ စကား ပြောပါ။",
    "နေကောင်း": "နေကောင်းပါတယ်ခင်ဗျာ။ ဘာများကူညီပေးရမလဲ။",
}


def chat_response(user_input):
    """Chat response using dataset data and fallback"""
    if not user_input or len(user_input.strip()) == 0:
        return "ပါသည်ကို ရေးပါ။"


    user_lower = user_input.lower()
    for pattern, response in chat_pairs:
        if pattern.lower() in user_lower or user_lower in pattern.lower():
            return response


    for key, response in fallback_responses.items():
        if key in user_lower:
            return response


    return "Amkyaw Ai သည် Myanmar NLP ပါဝင်သည့် အက်ပလီကေးရှင်း ဖြစ်ပါတယ်။ ရေးသားမှုနေပါသည်။"


def generate_text(prompt):
    """Simple text generation based on prompt"""
    if not prompt or len(prompt.strip()) == 0:
        return "ပါသည်ကို ရေးပါ။"
    if "သတင်း" in prompt:
        return "သတင်းသည်သင်တန်းစာသင်ပါး။ နိုင်ငံတော်သမိုင်းနဲ့ရေးသားပါ။ ကမ္ဘာနှင့် အလယ်ပိုင်းတွင် အရေးကြီးသည့် အချက်များ ပါဝင်ပါ။"
    elif "ကဗျာ" in prompt:
        return "မြန်မာကဗျာ ရေးလိုက်ပါ။ အိပ်မက်မှာ ပါ။ အကျယ်ပြန့်စွာ ရှိပါ။"
    elif "ဆေး" in prompt:
        return "ဆေးပပါး ဘာသာပါ။ ကျန်းမာရေးပါ။ ဆရာဝန်နဲ့ လမ်းညွှန်ပါ။"
    elif "ဥပဒေ" in prompt:
        return "ဥပဒေစာတွေ ရှိပါ။ စာခွင့် အရ လုပ်ပါ။"
    elif "ပညာ" in prompt:
        return "ပညာ သင်လိုက်ပါ။ သင်ယူခွင့် ပါ။ ကျောင်းသားများ ပါ။"
    elif "ဘာသာ" in prompt:
        return "ဘာသာစာသင်၊ ပါ။ သတင်း ပါ။ သံဃားစာပါ။"
    else:
        return "မြန်မာစာရေးသားပါ။ ရေးသားမှု လုပ်ပါ။"


# Simple classifier
def simple_classify(text):
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
    predicted = max(scores, key=scores.get) if max(scores.values()) > 0 else "other"
    return predicted


def predict(text):
    if not text or len(text.strip()) == 0:
        return {"error": "Please enter some text"}
    label = simple_classify(text)
    confidence = 0.85
    return {"text": text[:100] + "..." if len(text) > 100 else text, "label": label, "confidence": confidence}


# CSS Styles
css = """
.container {
    max-width: 1200px;
    margin: auto;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
    color: white;
    margin-bottom: 20px;
}
.tab-header {
    font-size: 18px;
    font-weight: bold;
}
.examples-box {
    background: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.button-primary {
    background: #667eea !important;
    color: white !important;
    border: none !important;
    padding: 10px 20px !important;
    border-radius: 5px !important;
}
.button-secondary {
    background: #764ba2 !important;
    color: white !important;
    border: none !important;
}
"""

# Gradio UI with better design
with gr.Blocks(title="AmkyawDev NLP", css=css, theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("""
    <div class="header">
        <h1>🇲🇲 AmkyawDev NLP</h1>
        <p>Myanmar Language AI - Classification, Chat & Text Generation</p>
        <p style="font-size: 14px;">Powered by AmkyawDev Dataset</p>
    </div>
    """)
    
    with gr.Tab("📊 Classification"):
        gr.Markdown("### 📊 Myanmar Text Classification")
        gr.Markdown("မြန်မာစာရေးသားပါ... ပါးပါး ခွဲခြားပါ။")
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="📝 Enter Myanmar Text",
                    placeholder="မြန်မာစာရေးသားပါ... (e.g., သတင်းသည်သင်တန်းစာသင်ပါး)",
                    lines=4,
                    show_label=True
                )
                submit_btn = gr.Button("🔍 Predict", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                output = gr.JSON(label="📊 Prediction Result")
        
        submit_btn.click(predict, input_text, output)
        
        # Examples Section
        gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                ["သတင်းသည်သင်တန်းစာသင်ပါး။"],
                ["မိတ်ဆွေများနဲ့ စကား ပြောပါ။"],
                ["ကဗျာ ရေးလိုက်ပါ။"],
                ["ဥပဒေစာတွေ ရှိပါ။"],
                ["ဆေးပပါး ဘာသာပါ။"],
                ["ပညာ သင်လိုက်ပါ။"],
            ],
            inputs=input_text,
        )
        
        # Info Box
        gr.Markdown("""
        <div style="background: #e8f4f8; padding: 15px; border-radius: 8px; margin-top: 10px;">
            <b>📌 Info:</b> ဤအပိုင်းသည် မြန်မာစာကို ခွဲခြားပါဝင်ပါ။ ဥပမာ - greeting, coding, culture, food, health, math, travel စသည်ဖြင့်။
        </div>
        """)
    
    with gr.Tab("💬 Chat"):
        gr.Markdown("### 💬 Myanmar Chat Bot")
        gr.Markdown("မြန်မာစာဖြင့် ပါးပါး ပါတ်ပါ။")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="💭 Chat History",
                    height=400,
                    show_label=True,
                    bubble_full_width=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("""
                <div style="background: #f0f0f0; padding: 15px; border-radius: 8px;">
                    <b>💡 Tips:</b><br>
                    - မြန်မာစာဖြင့် ရေးပါ။<br>
                    - ပါးပါး ပါတ်ပါ။
                </div>
                """)
        
        with gr.Row():
            with gr.Column(scale=3):
                msg = gr.Textbox(
                    label="✍️ Your Message",
                    placeholder="မြန်မာဘာသာဖြင့် ရေးပါ...",
                    lines=2,
                    show_label=True
                )
            with gr.Column(scale=1):
                with gr.Row():
                    send_btn = gr.Button("📤 Send", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear")
        
        def respond(message, history):
            if not message:
                return "", history or []
            response = chat_response(message)
            if history is None:
                history = []
            history.append({"role": "user", "content": [{"text": message, "type": "text"}]})
            history.append({"role": "assistant", "content": [{"text": response, "type": "text"}]})
            return "", history
        
        send_btn.click(respond, [msg, chatbot], [msg, chatbot])
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear_btn.click(lambda: [], None, chatbot)
        
        # Chat Examples
        gr.Markdown("### 💡 Quick Questions")
        gr.Examples(
            examples=[
                ["နေကောင်းလား?"],
                ["သတင်းပါသည်။"],
                ["ကဗျာ ရေးပါ။"],
                ["ဆေးကုသပါ။"],
                ["ဥပဒေ ပါ။"],
                ["ပညာ သင်ပါ။"],
            ],
            inputs=msg,
        )
    
    with gr.Tab("✍️ Text Generate"):
        gr.Markdown("### ✍️ Myanmar Text Generation")
        gr.Markdown("မြန်မာစာ ထုတ်လုပ်ပါ။")
        
        with gr.Row():
            with gr.Column(scale=1):
                gen_prompt = gr.Textbox(
                    label="📝 Prompt (အစ)",
                    placeholder="သတင်း... သို့မဟုတ် ကဗျာ...",
                    lines=4,
                    show_label=True
                )
                gen_btn = gr.Button("✨ Generate", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gen_output = gr.Textbox(
                    label="📄 Generated Text",
                    lines=6,
                    interactive=False,
                    show_label=True
                )
        
        gen_btn.click(generate_text, gen_prompt, gen_output)
        gen_prompt.submit(generate_text, gen_prompt, gen_output)
        
        # Examples
        gr.Markdown("### 💡 Examples")
        gr.Examples(
            examples=[
                ["သတင်း ဖတ်ပါ။"],
                ["ကဗျာ ရေးပါ။"],
                ["ဆေးကုသပါ။"],
                ["ဥပဒေ ပါ။"],
                ["ပညာ သင်ပါ။"],
                ["ဘာသာ ပါ။"],
            ],
            inputs=gen_prompt,
        )
        
        # Info Box
        gr.Markdown("""
        <div style="background: #f0e8ff; padding: 15px; border-radius: 8px; margin-top: 10px;">
            <b>📌 Info:</b> ဤအပိုင်းသည် မြန်မာစာ ထုတ်လုပ်ပါဝင်ပါ။ Prompt ပါးပါး ရေးပါ။
        </div>
        """)
    
    # Footer
    gr.Markdown("""
    ---
    ### 📚 Resources
    - **Dataset:** [AmkyawDev-Dataset](https://huggingface.co/datasets/amkyawdev/AmkyawDev-Dataset)
    - **GitHub:** [amkyawdev/myanmar-lnp-dataset](https://github.com/amkyawdev/myanmar-lnp-dataset)
    - **Space:** [amkyawdev-nlp](https://huggingface.co/spaces/amkyawdev/amkyawdev-nlp)
    ---
    Made with ❤️ by AmkyawDev
    """)


if __name__ == "__main__":
    demo.launch()
