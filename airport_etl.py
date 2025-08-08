import random
from datetime import datetime, timedelta
import os

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # pragma: no cover - basic loader
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as env_file:
            for line in env_file:
                if line.strip() and not line.startswith("#"):
                    key, _, value = line.strip().partition("=")
                    os.environ.setdefault(key, value)

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, count
    SPARK_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when pyspark is missing
    SPARK_AVAILABLE = False

try:  # pragma: no cover - langchain is optional
    from langchain.agents import initialize_agent, Tool, AgentType
    from langchain.chat_models import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except Exception:  # pragma: no cover
    LANGCHAIN_AVAILABLE = False


def generate_simulated_data(spark=None):
    """Generate simulated airport flight data."""
    airlines = ["TAM", "GOL", "AZUL"]
    data = []
    base_time = datetime(2024, 1, 1, 6, 0)
    for i in range(10):
        dep = base_time + timedelta(minutes=10 * i)
        arr = dep + timedelta(minutes=60 + random.randint(-10, 20))
        delay = random.randint(-15, 120)
        passengers = random.randint(50, 300)
        data.append({
            "flight_id": f"FL{i:03d}",
            "airline": random.choice(airlines),
            "departure": dep.isoformat(),
            "arrival": arr.isoformat(),
            "delay": delay,
            "passengers": passengers,
        })
    if spark:
        return spark.createDataFrame(data)
    return data


def transform_data(data, spark=None, limit=1000):
    """Compute statistics using Spark when available."""
    if spark:
        df = data
        stats = (
            df.groupBy("airline")
            .agg(avg("delay").alias("avg_delay"), count("*").alias("num_flights"))
            .orderBy("airline")
        )
        # Avoid bringing a very large result back to the driver
        limited = stats.limit(limit)
        limited.write.mode("overwrite").parquet("output/stats")
        return [row.asDict() for row in limited.collect()]

    # fallback plain Python implementation
    totals = {}
    for row in data:
        a = row["airline"]
        totals.setdefault(a, {"delay": 0, "flights": 0})
        totals[a]["delay"] += row["delay"]
        totals[a]["flights"] += 1
    results = []
    for a, val in sorted(totals.items()):
        results.append(
            {
                "airline": a,
                "avg_delay": val["delay"] / val["flights"],
                "num_flights": val["flights"],
            }
        )
    return results


class SimpleReactAgent:
    """Very small ReAct-style agent executed localmente."""

    def __init__(self, persona: str = "analista de pontualidade"):
        self.persona = persona

    def analyze(self, stats):
        thoughts = [f"Persona: {self.persona}"]
        thoughts.append("Thought: preciso analisar o atraso médio por companhia.")
        lines = [
            f"  - {s['airline']}: atraso médio {s['avg_delay']:.1f} min em {s['num_flights']} voos"
            for s in stats
        ]
        thoughts.append("Action: observo as estatísticas calculadas.")
        thoughts.extend(["Observation: veja os dados:"] + lines)
        worst = max(stats, key=lambda s: s["avg_delay"])
        thoughts.append(
            f"Thought: a companhia com maior atraso médio é {worst['airline']}.")
        thoughts.append(
            f"Final Answer: {worst['airline']} possui o maior atraso médio ({worst['avg_delay']:.1f} min).")
        return "\n".join(thoughts)


class LangChainReactAgent:
    """ReAct agent powered by LangChain and OpenAI."""

    def __init__(self, persona: str = "analista de pontualidade"):
        self.persona = persona

    def analyze(self, stats):  # pragma: no cover - depends on external service
        llm = ChatOpenAI(temperature=0)
        tool = Tool(
            name="flight_stats",
            func=lambda _: str(stats),
            description="Estatísticas de atraso médio por companhia aérea",
        )
        agent = initialize_agent(
            [tool],
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={
                "prefix": f"Você é {self.persona} e deve responder em português usando o estilo ReAct."
            },
        )
        prompt = "Qual companhia aérea tem o maior atraso médio?"
        return agent.run(prompt)


def main():
    spark = None
    if SPARK_AVAILABLE:
        spark = (
            SparkSession.builder.master("local[*]")
            .appName("AirportETL")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
    data = generate_simulated_data(spark)
    stats = transform_data(data, spark)

    personas = [
        p.strip()
        for p in os.getenv("AGENT_PERSONAS", "analista de pontualidade").split(",")
        if p.strip()
    ]

    analyses = []
    for persona in personas:
        if LANGCHAIN_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                agent = LangChainReactAgent(persona)
                analysis = agent.analyze(stats)
            except Exception:
                agent = SimpleReactAgent(persona)
                analysis = agent.analyze(stats)
        else:
            agent = SimpleReactAgent(persona)
            analysis = agent.analyze(stats)
        analyses.append(analysis)

    with open("analysis_result.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(analyses) + "\n")
    if spark:
        spark.stop()


if __name__ == "__main__":
    main()
