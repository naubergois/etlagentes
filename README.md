# ETL de Aeroporto com Agente ReAct

Este projeto demonstra um pipeline ETL simples usando dados simulados de um aeroporto. O fluxo realiza:

1. **Extração e geração** de voos fictícios.
2. **Transformação** e cálculo de estatísticas com Apache Spark quando disponível (há implementação em Python puro como fallback).
3. **Análise** por agentes estilo *ReAct*. As personas dos agentes podem ser definidas via `AGENT_PERSONAS` e cada persona gera sua própria conclusão. Se [LangChain](https://python.langchain.com/) e os modelos da OpenAI estiverem configurados, a análise usa LLM; caso contrário, há um agente local simples.
4. **Carga** do resultado final em `analysis_result.txt`.

## Requisitos

- Python 3.9+
- (Opcional) [PySpark](https://spark.apache.org/docs/latest/api/python/) para usar o Spark.
- (Opcional) [LangChain](https://python.langchain.com/), [OpenAI](https://pypi.org/project/openai/) e [python-dotenv](https://pypi.org/project/python-dotenv/) para o agente baseado em LLM.

Para instalar as dependências opcionais quando houver acesso à internet:

```bash
pip install pyspark langchain openai python-dotenv
```

## Execução

```bash
python airport_etl.py
```

O script gera o arquivo `analysis_result.txt` com as observações de cada persona.

## Estrutura do Projeto

- `airport_etl.py` – script com geração de dados, transformações e agentes de análise.
- `analysis_result.txt` – exemplo de saída com a análise produzida.
- `.env.example` – modelo de arquivo para definir `OPENAI_API_KEY` e `AGENT_PERSONAS`.

Para usar o agente baseado em LLM, copie `.env.example` para `.env` e informe sua chave da OpenAI:

```bash
cp .env.example .env
# edite .env e insira sua chave
```

No mesmo arquivo `.env`, opcionalmente defina `AGENT_PERSONAS` com uma lista separada por vírgula para executar múltiplas personas:

```bash
AGENT_PERSONAS="analista de pontualidade,gerente de operações"
```

