import ollama
from ollama import ChatResponse
from typing import Optional, Any, Dict



class Ollama:

    def __init__(self, model_name: str, model_params: Optional[Dict[str, Any]]=None):
        """
        Inizializza la classe OllamaLLM con il modello specificato.

        :param model_name: Il nome del modello da utilizzare.
        :param model_params: Set di parametri per configuarea il modello invocato. Di default è None
        """
        self.model_name=model_name
        self.model_params=model_params

    def generate(self, query: str, context: str) -> ChatResponse:

        response: ChatResponse = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": context
                },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ]
        )
        return response
