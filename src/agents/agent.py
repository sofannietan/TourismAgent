from langchain.agents import create_agent
from langchain_classic import hub
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

def build_agent(llm, tools:list, callbacks:list):
    template="""Answer the following questions as best as you can? You have acces to the following tools : {tools}.
    Use the following format : 
    Question : the input question you must answer.
    Thought : you should always think about what you do.
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation : the result of the action
    .... (this Thought/action/action input/observation can repeat N times)
    Thought : I now know the final answer.
    Final answer : the final answer to the input question
    Begin !
    Question {input}
    Thought : {agent_scratchpad}"""
    agent = create_agent(model = llm,
                         tools = tools,
                         system_prompt= template)
    return agent

def build_query_router(llm):
    class QueryRouter(BaseModel):
        route : Literal["rag","agent"]=Field(description=" Choose 'rag' if the question is on information on history, culture, internal notes, textual description, temple description, facts, travel. Choose 'agent'for currency calculation, to pick the best season to travel or for the weather")
        reasoning: str= Field(description ="Explain in detail your decision" )
    router_llm = llm.with_structured_output(QueryRouter)
    router_prompt=ChatPromptTemplate.from_template("""
                                                   Analyze the question and decide whether choose 'rag' or 'agent'. CHoose 'rag' if the question is on information on history, culture, internal notes, textual description, temple description, facts travel. Choose 'agent'for currency calculation, to pick the best season to travel or for the weather
                                                   Question : {question}""")
    router_chain = router_prompt | router_llm
    return router_chain

def router_query(question:str, router)-> str:
    result= router.invoke({"question": question})
    return result
    