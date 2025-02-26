import os
import dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# load the environment variables
dotenv.load_dotenv()

# instanciate the FastAPI class
app = FastAPI(
    title="Agente de IA",
    description="""
        Mi primer bot básico
        1. * Consultas a un modelo cohere
    """,
    version="v0.1.0"
)


#model schema
class Agente(BaseModel):
    prompt: str

@app.post("/agente/")
async def agente(request: Agente):

    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    prompt=request.prompt

    llm = ChatCohere( 
        api_key=COHERE_API_KEY
    )

    list_message = [
        SystemMessage(content="Eres experto en IA, te llamas PEPITO."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(list_message)
    list_message.append(response)
    
    return {
        "agente": list_message[-1].content,
    }
