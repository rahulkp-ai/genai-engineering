# Generative AI Engineering

> Building LLM systems from first principles — fine-tuning, RAG pipelines,
> NLP architectures, and production-grade AI applications.

## 🗂 What's Here

| Area                                      | Focus                               | Status      |
| ----------------------------------------- | ----------------------------------- | ----------- |
| [Deep Learning](./deep-learning/)         | CNNs, RNNs, Transformers in PyTorch | ✅ Active   |
| [GenAI Engineering](./genai-engineering/) | LoRA fine-tuning, RLHF, DPO         | ✅ Active   |
| [RAG & Agents](./rag-and-agents/)         | LangChain, vector DBs, AI agents    | 🔨 Building |
| [Projects](./projects/)                   | Original mini-projects              | 🔨 Building |

## 🔬 Technical Stack

**Core:** Python 3.11 · PyTorch 2.x · Hugging Face Transformers  
**GenAI:** LangChain · OpenAI API · Gradio · FAISS / ChromaDB  
**ML:** scikit-learn · Keras · NumPy · Pandas  
**Dev:** pytest · black · ruff · Git

## 🧠 Conceptual Foundation

→ [`ann-foundation`](https://github.com/rahulkp-ai/ann-foundation) —
Autograd engine and MLP built from scratch. The mathematical bedrock for everything in this repo.

## 📌 Featured Work

- **[LoRA Fine-tuning Pipeline](./genai-engineering/fine-tuning/)** — Parameter-efficient fine-tuning with configurable rank, target modules, and training logging
- **[RAG Question-Answering System](./rag-and-agents/rag-system/)** — Document ingestion → chunking → embedding → retrieval → generation pipeline
- **[Transformer from Scratch](./deep-learning/transformers/attention_mechanism.py)** — Attention, positional encoding, and masking implemented in pure PyTorch

## 🛠 Setup

```bash
git clone https://github.com/rahulkp-ai/genai-engineering
cd genai-engineering
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

_Actively building. New work added weekly._
