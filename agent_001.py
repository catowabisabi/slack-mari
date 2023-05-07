"""
This agent do google search and give you answer acording to the result.
"""
from common_imports.common_imports import * # all the key are here
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType

class ChatAgent:
    def run(self, question):
        
        try:
            llm = ChatOpenAI(temperature=0.5)
            math_llm = OpenAI(temperature=0.0)
            tools = load_tools(["serpapi", "llm-math", "human"], llm=llm)
            tools = load_tools(
            ["serpapi", "human", "llm-math"], 
            llm=math_llm,
        )
            agent_chain = initialize_agent(
            tools,
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )
            agent_chain.run(question)

        except Exception as e:
            print("\n\nAgent 001 / Starting ... Error: \n" + str(e) + "\n\n")

class SearchAgent:
    def run(self, question):
        llm = OpenAI(temperature=0)
        tools_search = load_tools(["serpapi", "llm-math"], llm=llm)
        agent = initialize_agent(tools_search, 
                            llm, 
                            agent="zero-shot-react-description", 
                            verbose=True)
        agent.run(question)

class WikiAgent:
    def run(self, question):
        llm = OpenAI(temperature=0)
        wiki_tools = load_tools(["serpapi", "llm-math", "wikipedia", "terminal"], llm=llm)
        agent = initialize_agent(wiki_tools, 
                            llm, 
                            agent="zero-shot-react-description", 
                            verbose=True)
        agent.agent.llm_chain.prompt.template
        agent.run(question)

wa = WikiAgent()
while True:
    question = input("What is your question? ")
    wa.run(question)