import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
from serpapi.google_search import GoogleSearch
import nest_asyncio
import datetime
import sqlite3

# Carregar variáveis de ambiente
load_dotenv()
nest_asyncio.apply()

# Configurar chaves de API
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Função para buscar insights na web via SerpAPI
def fetch_insights(query: str, k: int = 5) -> str:
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_KEY,
        "num": k
    }
    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    snippets = [r.get("snippet", "") for r in results]
    return "\n".join(snippets)

# Inicializar LLM da OpenAI
llm = OpenAI(temperature=0.7, model_name="gpt-4o")

# Definir agentes
planner = Agent(
    role="Planejador de Conteúdo",
    goal="Planejar conteúdo envolvente e preciso para redes sociais sobre {topic}",
    backstory=(
        "Você está planejando um post para redes sociais sobre o tópico: {topic}. "
        "Use os seguintes insights da web para embasar seu plano:\n{web_insights}\n"
        "Seu trabalho é a base para o Criador de Conteúdo criar o post em redes sociais."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

creator = Agent(
    role="Criador de Conteúdo",
    goal="Criar um post envolvente e bem estruturado para redes sociais sobre {topic}.",
    backstory=(
        "Você é um criador de conteúdo que usa o plano para criar um post para redes sociais. "
        "Certifique-se de alinhar com práticas de copywriting e com a voz da marca."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Editar o post gerado para alinhamento com a voz e estilo da marca.",
    backstory=(
        "Você é um editor que recebe um post em redes sociais criado. "
        "Revise para seguir boas práticas, tom de voz e sem erros."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# Definir tarefas
plan = Task(
    description=(
        "1. Priorizar tendências recentes, principais atores e notícias relevantes sobre {topic}.\n"
        "2. Use estes insights da web:\n{web_insights}\n"
        "3. Identificar o público-alvo e seus pontos de dor.\n"
        "4. Desenvolver um esboço detalhado com hook, corpo e call to action.\n"
        "5. Incluir hashtags e palavras-chave relevantes.\n"
        "6. O plano deve cobrir todos os aspectos para um post em redes sociais."
    ),
    expected_output="Plano de conteúdo com outline, público, hashtags e CTA.",
    agent=planner,
)

create = Task(
    description=(
        "1. Use o plano de conteúdo para criar um post para redes sociais sobre {topic}.\n"
        "2. Incorpore hashtags de forma natural.\n"
        "3. Estruture com hook inicial, corpo informativo e chamada para ação.\n"
        "4. Mantenha o texto conciso (até 300 caracteres) e cativante.\n"
        "5. Revise alinhamento com a voz da marca."
    ),
    expected_output="Texto de post em redes sociais, pronto para publicação.",
    agent=creator,
)

edit = Task(
    description="Revise o post para erros gramaticais e alinhe com a voz da marca.",
    expected_output="Post revisado e sem erros para redes sociais.",
    agent=editor
)

# Persistência local (SQLite)
conn = sqlite3.connect("content.db", check_same_thread=False)
c = conn.cursor()
c.execute(
    """
    CREATE TABLE IF NOT EXISTS posts(
        id INTEGER PRIMARY KEY,
        topic TEXT,
        date TEXT,
        content TEXT
    )
    """
)
conn.commit()

def save_to_db(topic: str, content: str):
    today = datetime.date.today().isoformat()
    c.execute(
        "INSERT INTO posts(topic, date, content) VALUES (?, ?, ?)",
        (topic, today, content)
    )
    conn.commit()

def save_to_file(topic: str, content: str) -> str:
    hoje = datetime.date.today().isoformat()
    pasta = "posts"
    os.makedirs(pasta, exist_ok=True)
    nome = f"{hoje}_{topic.replace(' ', '_')}.md"
    caminho = os.path.join(pasta, nome)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write(content)
    return caminho

# Configurar Crew
crew = Crew(
    agents=[planner, creator, editor],
    tasks=[plan, create, edit],
    verbose=True
)

# Aplicativo Streamlit
def main():
    st.title("Assistente de Criação de Conteúdo para Redes Sociais")

    # CRUD na sidebar
    st.sidebar.title("Histórico de Posts")
    rows = c.execute("SELECT id, topic, date FROM posts").fetchall()
    options = [""] + [f"{r[0]} – {r[1]} ({r[2]})" for r in rows]
    sel = st.sidebar.selectbox("Selecione um post", options)
    if sel:
        pid = int(sel.split(" – ")[0])
        content_db = c.execute("SELECT content FROM posts WHERE id = ?", (pid,)).fetchone()[0]
        st.sidebar.markdown("### Conteúdo")
        st.sidebar.write(content_db)
        if st.sidebar.button("❌ Deletar"):
            c.execute("DELETE FROM posts WHERE id = ?", (pid,))
            conn.commit()
            st.sidebar.success("Post deletado!")

    topic = st.text_input("Digite o tópico para o post:")
    framework = st.selectbox("Framework de Copy", ["AIDA", "PAS"])

    if st.button("Gerar Conteúdo"):
        if topic:
            with st.spinner("Buscando insights na web..."):
                web_insights = fetch_insights(topic)
            with st.spinner("Gerando conteúdo..."):
                raw = crew.kickoff(inputs={"topic": topic, "web_insights": web_insights})
                result = str(raw)
                st.markdown(result)
                save_to_db(topic, result)
                path = save_to_file(topic, result)
                st.success(f"📝 Post salvo em: {path}")

                # Botão de variação
                st.session_state["last_post"] = result
                if st.button("🔄 Gerar variação"):
                    prompt_var = (
                        "Gere uma variação do post abaixo, mantendo tom e extensão:\n\n"
                        + st.session_state["last_post"]
                    )
                    variation = llm(prompt_var)
                    st.markdown(variation)
        else:
            st.error("Por favor, digite um tópico.")

if __name__ == "__main__":
    main()
