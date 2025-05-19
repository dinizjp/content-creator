import streamlit as st
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain.llms import OpenAI
import nest_asyncio

# Carregar variáveis de ambiente
load_dotenv()
nest_asyncio.apply()

# Configurar chave da OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Inicializar LLM da OpenAI
llm = OpenAI(temperature=0.7, model_name="gpt-4o-mini")

# Definir agentes
planner = Agent(
    role="Planejador de Conteúdo",
    goal="Planejar conteúdo envolvente e preciso sobre {topic}",
    backstory=(
        "Você está planejando um artigo de blog sobre o tópico: {topic}. "
        "Você coleta informações que ajudem a audiência a aprender e tomar decisões informadas. "
        "Seu trabalho é a base para o Escritor de Conteúdo criar o post."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Escritor de Conteúdo",
    goal="Escrever um post de blog envolvente e bem estruturado sobre {topic}.",
    backstory=(
        "Você é um escritor que usa o plano de conteúdo para criar um post detalhado. "
        "Certifique-se de alinhar com práticas de SEO e com a voz da marca."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Editar um post de blog para alinhar ao estilo da organização.",
    backstory=(
        "Você é um editor que recebe um post do Escritor de Conteúdo. "
        "Revise para seguir práticas jornalísticas, fornecer pontos de vista balanceados e evitar temas controversos."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# Definir tarefas
plan = Task(
    description=(
        "1. Priorizar tendências recentes, principais atores e notícias relevantes sobre {topic}.\n"
        "2. Identificar o público-alvo, considerando interesses e pontos de dor.\n"
        "3. Desenvolver um esboço detalhado com introdução, pontos-chave e call to action.\n"
        "4. Incluir palavras-chave de SEO e fontes relevantes.\n"
        "5. O plano deve ser abrangente e cobrir todos os aspectos do tópico."
    ),
    expected_output="Documento de plano de conteúdo com esboço, análise de público, palavras-chave de SEO e recursos.",
    agent=planner,
)

write = Task(
    description=(
        "1. Use o plano de conteúdo para criar um post envolvente sobre {topic}.\n"
        "2. Incorpore palavras-chave de SEO de forma natural.\n"
        "3. Seções/subtítulos nomeados de forma envolvente.\n"
        "4. Estrutura com introdução atraente, corpo informativo e conclusão resumida.\n"
        "5. Revise erros gramaticais e garanta alinhamento com a voz da marca.\n"
        "6. Cada seção deve ter 2 ou 3 parágrafos.\n"
        "7. O post completo deve ter pelo menos 1000 palavras."
    ),
    expected_output="Post de blog em markdown, polido e pronto para publicação.",
    agent=writer,
)

edit = Task(
    description="Revise o post para erros gramaticais e alinhamento com a voz da marca.",
    expected_output="Post revisado e sem erros em markdown, pronto para publicação.",
    agent=editor
)

# Configurar Crew
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# Aplicativo Streamlit
def main():
    st.title("Assistente de Criação de Conteúdo")

    topic = st.text_input("Digite o tópico para o post:")
    if st.button("Gerar Conteúdo"):
        if topic:
            with st.spinner("Gerando conteúdo..."):
                try:
                    result = crew.kickoff(inputs={"topic": topic})
                    st.markdown(result)
                except Exception as e:
                    st.error(f"Ocorreu um erro: {e}")
        else:
            st.error("Por favor, digite um tópico.")

if __name__ == "__main__":
    main()
