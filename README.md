
<div align="center">

# 🤖 Open Source LLM Chatbot

A professional-tier implementation of conversational AI agents using open-source Large Language Models (LLMs) and Hugging Face Transformers.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-success?style=for-the-badge)](#)
[![IBM Certification](https://img.shields.io/badge/IBM-AI%20Developer%20Program-blue?style=for-the-badge&logo=ibm)](https://cognitiveclass.ai/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](#)

</div>

---

## 📌 Project Overview

This repository demonstrates the evolution of chatbot architectures: transitioning from traditional sequence-to-sequence (Seq2Seq) model implementations to modern causal LLMs utilizing structured chat templates.

It contains two complete conversational AI implementations built with Python and PyTorch:

* **Legacy Seq2Seq Engine (`chatbot.py`):** Uses Facebook's `blenderbot-400M-distill`. Highlights foundational NLP principles including manual prompt formatting, tokenization, sequence-to-sequence generation, detokenization, and state-managed context retention.
* **Modern Causal LLM Engine (`chatbot_llm.py`):** Uses Hugging Face's instruction-tuned `SmolLM2-360M-Instruct`. Implements state-of-the-art Hugging Face Chat Templates (`apply_chat_template`), role-based messaging (system, user, assistant), and memory-efficient inference execution (`torch.inference_mode`).

---

## 🏗️ Architecture Flow Diagram


```mermaid
graph TD
    User(["👤 User Input (Terminal)"]) --> History[("📂 Conversation History Buffer")]
    
    History -->|"Maintains rolling context window (System Prompt + Last N Turns)"| Template["⚙️ Prompt Formatting & Chat Template"]
    
    Template -->|"Translates structured roles into model-specific tokens"| Generation["🧠 Model Generation (PyTorch)"]
    
    Generation -->|"Generates response tokens using top-p & temperature"| Detokenization["📝 Detokenization & Slicing Output"]
    
    Detokenization -->|"Decodes token IDs & appends to state buffer"| History
    
    Detokenization --> Bot(["🤖 Bot Response (Terminal)"])
    
    style User fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
    style History fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000
    style Template fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000
    style Generation fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000
    style Detokenization fill:#f3e5f5,stroke:#8e24aa,stroke-width:2px,color:#000
    style Bot fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000
```
---

## ✨ Key Features

* **Dual Architecture Paradigms**: Demonstrates hands-on knowledge of both Seq2Seq and modern Causal/Decoder-only transformer architectures.
* **Hugging Face Chat Templates**: Utilizes standardized formatting (`apply_chat_template`) to manage multi-turn role interactions seamlessly. 
* **Rolling Context Management**: Features automatic window trimming to maintain context within maximum token limits without overflowing model memory.
* **CPU-Optimized Inference**: Implements `torch.inference_mode()` with lightweight model variants for rapid execution without needing high-end GPU resources. 
* **Hyperparameter Tuning**: Fine-tuned output generation settings including `temperature`, `top_p`, `repetition_penalty`, and `no_repeat_ngram_size` to control creative variance and minimize repetitive text loops.

---

## 🛠️ Core Tech Stack

| Technology | Purpose | Version |
| :--- | :--- | :--- |
| **Python** | Primary Programming Language | `3.10+` |
| **Hugging Face Transformers** | Model Architecture & Tokenizer Interfaces | `4.41.2` |
| **PyTorch** | Deep Learning Framework & Tensor Processing | `2.2.2` |
| **Accelerate** | Efficient Model Loading & Execution | `0.30.1` |
| **NumPy** | High-Performance Array Operations | `1.26.4` |

---

## 📁 Repository Structure
```text
open-source-llm-chatbot/
├── .theia/
│   └── settings.json          # IDE workspace configurations
├── .gitignore                 # Specifies intentionally untracked virtual environments
├── requirements.txt           # Pinned dependencies for reproducible execution
├── chatbot.py                 # Implementation 1: Seq2Seq BlenderBot Model
├── chatbot_llm.py             # Implementation 2: Modern Causal SmolLM2 Chat Engine
└── README.md                  # Detailed project documentation
```
---

# 🚀 Local Setup & Execution

### Prerequisites
* Python 3.10 or higher installed.
* `git` CLI tool configured on your machine.

1. Clone the Repository
```bash
git clone [https://github.com/HAMED-PAYANDA/open-source-llm-chatbot.git](https://github.com/HAMED-PAYANDA/open-source-llm-chatbot.git)
cd open-source-llm-chatbot
```
2. Create and Activate Virtual Environment
# Install virtualenv package if needed
```bash
pip3 install virtualenv

# Create environment
virtualenv my_env

# Activate environment
# On Linux/macOS:
source my_env/bin/activate
# On Windows Command Prompt:
# my_env\Scripts\activate.bat
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Running the Chatbots
You can choose to run either implementation. To terminate either session, type exit at the command prompt.
Option A: Run the Modern Causal LLM Engine (Recommended)
```bash
python3 chatbot_llm.py
```
![Screenshot of the terminal showing the Hugging Face model loading and answering a user prompt](screenshot3.png)

Option B: Run the Legacy Seq2Seq Engine
```bash
python3 chatbot.py
```
---

## 📜 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## 👤 Author

**Hamed Payanda**
* **GitHub:** [@HAMED-PAYANDA](https://github.com/HAMED-PAYANDA)
* Completed as part of the **IBM AI Developer Program**.



