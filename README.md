# Language Models and Structured Data (APM_5AI29_TP) - 2024/2025

## Course Overview

This repository contains materials and resources for the course **APM_5AI29_TP: Language Models and Structured Data**, part of the **Data and Artificial Intelligence** curriculum. The course explores the use of language models beyond traditional NLP tasks, applying them to structured data formats such as graphs, tables, and databases. Topics include prompt engineering, retrieval augmented generation, and code generation.

### Key Topics:

1. Language Modeling and Prompt Engineering – Understanding the foundations of language models and their interaction with prompts.
2. Retrieval Augmented Generation (RAG) – Techniques to improve model responses by integrating external knowledge sources.
3. Language Models on Graphs – Application of LLMs to tasks involving graph completion and node classification.
4. Language Models on Tabular Data & Databases – Interpreting and generating insights from tabular data using language models.
5. Text to Query Language (SQL) – Converting natural language questions into SQL queries.
6. Code Generation – Using language models to generate code for various programming tasks.

## Prerequisites

Students are expected to have:
- Basic knowledge of machine learning and NLP.
- Familiarity with Python programming and basic SQL.

## Course Structure

- Total Hours: 24 hours of in-person sessions (16 sessions).
- Credits:
  - M2 DATAAI: 2.5 ECTS
  - Diplôme d'ingénieur: 2 ECTS
  - Exchange/International Programs: 2 ECTS
- Evaluation: Grading will be based on a project carried out throughout the course and practical lab sessions.

## Instructor

- Professor Mehwish Alam

## Installation and Setup

For practical exercises, you will need Python, PyTorch, and libraries for NLP and structured data handling. Follow the instructions below to set up your environment using `conda`:

1. Create the Environment:
   ```bash
   conda create -n lm-structured python matplotlib numpy scipy scikit-image ipykernel pandas scikit-learn jupyter tqdm graphdatascience langchain langchain-core langchain-community pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -c conda-forge
   ```
2. Activate the Environment:
   ```bash
   conda activate lm-structured
   ```
3. Install ollama and PyKEEN:
   ```bash
   pip install ollama pykeen
   ```
4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

This setup will allow you to implement and test language models for structured data, such as querying databases, and completing graphs.

## How to Contribute

Feel free to contribute to the repository by:
- Submitting pull requests for corrections or improvements.
- Adding new examples or expanding existing projects.
