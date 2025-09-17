# LLM PMID Checker

A system for checking whether research triples are supported by PubMed abstracts using large language models.

## Overview

This system checks whether a given research triple (e.g., `['SIX1', 'affects', 'Cell Proliferation']`) is supported by a list of PubMed IDs (PMIDs). It:

1. **Normalizes entities** using ARAX and Node Normalization APIs to find semantic synonyms
2. **Extracts abstracts** from PMIDs using NCBI E-utilities
3. **Checks support** using local LLM inference via Ollama
4. **Reports results** with confidence scores and supporting sentences

## Quick Start

### 1. Install Dependencies

If you haven't pre-installed Ollama, you can download and install it from [here](https://ollama.com/download).

```bash
conda activate llm_pmid_env
pip install -r requirements.txt
```

### 2. Setup Ollama

Run the provided setup script:

```bash
pkill -f "ollama serve"
./setup_ollama.sh
```

This bash script will automatically launch Ollama and pull the necessary models. However, the hermes4 model is not available in the Ollama model hub yet, so you will need to manually install this model from Hugging Face. You can simply run the script `manual_install_hermes4.sh` to install the model.

```bash
./manual_install_hermes4.sh
```

### 3. Configure Environment

Create a `.env` file:

```bash
# Ollama Configuration  
OLLAMA_BASE_URL=http://localhost:11434

# HERMES Configuration
HERMES_MODEL=hermes4:70b

# GPT-OSS Configuration
GPT_OSS_MODEL=gpt-oss:120b
GPT_OSS_REASONING=high
GPT_OSS_SIZE=120b

# NCBI E-utilities Configuration
NCBI_EMAIL=your.email@example.com
NCBI_API_KEY=your_ncbi_api_key_here

# ARAX and Node Normalization APIs
# ARAX_BASE_URL=https://arax.transltr.io/api/arax/v1.4
# NODE_NORM_BASE_URL=https://nodenormalization-sri.renci.org
```

### 4. Run Checker

#### Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` and use the interactive interface to:
- Enter research triples to check
- Input PMIDs manually or via file upload  
- Select AI models
- View results with charts and export options

#### Command Line Interface

**Using Entity Names:**
```bash
# Using GPT-OSS 120B
python main.py --model gpt-oss --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929 16488997

# Using Hermes 4 70B
python main.py --model hermes4 --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids 34513929
```

**Using CURIEs:**
```bash
# Using GPT-OSS 120B
python main.py --model gpt-oss --triple_curie 'NCBIGene:6495' 'affects' 'UMLS:C0596290' --pmids 34513929 16488997

# Using Hermes 4 70B with file input
python main.py --model hermes4 --triple_name 'SIX1' 'affects' 'Cell Proliferation' --pmids-file pmids.txt
```

## Available Models

- **Hermes 4 70B**: Latest model by Nous Research with hybrid reasoning mode, Q4_K_XL quantization (~42GB VRAM)
- **GPT-OSS 120B**: Model developed by OpenAI (~65GB VRAM)
