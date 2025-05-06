Mosscap: A local interaction
============================

Local LLM chat using Streamlit with LangChain and Ollama.

Goals:

- Allow user to pick models downloaded via Ollama
- *Remember* chat history during session
- Store conversations for reference in future sessions


Install an Ollama model:
------------------------

Ollama install:
---------------

```
>> apt update
>> apt install -y curl
>> curl -fsSL https://ollama.com/install.sh | sh
```

Once installed, confirm...

```
>> ollama --version
>> ollama --help
```

Pull an Ollama Model
--------------------

Ollama bundles model weights and runtime, your just pull the model you need:

```
>> ollama pull mistral
>> ollama pull gemma3:12b-it-qat
```

(*You can find a list of models at https://ollama.com/models*)

You can test that the model pulled with the following:

```
>> ollama list
>> ollama show mistral
>> ollama run mistral "Can you tell me about RAG please?"
```

Run the Chat:
-------------

```
>> streamlit run mosscap_chat.py -- --llm_model gemma3:12b-it-qat
```
