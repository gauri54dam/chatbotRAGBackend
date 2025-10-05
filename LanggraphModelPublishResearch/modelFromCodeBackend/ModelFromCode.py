# Databricks notebook source
## https://mlflow.org/blog/langgraph-model-from-code
# MAGIC %md
# MAGIC ### Install modules

# COMMAND ----------

# MAGIC %pip install mlflow langchain databricks-vectorsearch databricks-sdk mlflow[databricks] dotenv langgraph langchain_community langchain-core

# COMMAND ----------

!pip install IPython

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

!python --version

# COMMAND ----------

from dotenv import load_dotenv
load_dotenv()
import os
os.environ["DATABRICKS_TOKEN"] = os.getenv("DATABRICKS_TOKEN")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Compile Langgraph

# COMMAND ----------

## basic graph
!python ./graph.py

# COMMAND ----------

from graph import load_graph

#Load the graph
graph = load_graph()

# COMMAND ----------

graph

# COMMAND ----------

graph.nodes

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Publish

# COMMAND ----------

# MLFLow -------------
from mlflow.models import infer_signature
import mlflow
import langchain
import langchain_core
import langchain_community


# COMMAND ----------

## 1. Normal HR question
## 2. HR question with factual enhancement from internet
example_input = {
    "messages": [{"role": "user", "content": "How do I do expenses when I travel?"}],
    "messages": [{"role": "user", "content": "What are 2025 holidays list alongwith its calender dates in year 2025?"}]
}

example_output = {
        "messages": [{"role": "assistant", "content": 
        "To do expenses when you travel, you must submit them in a timely manner with proper documentation, including original receipts for expenditures over $25, and ensure they are reasonable and directly related to company business. All expenses must be pre-approved by your manager or project lead and adhere to the company's Travel and Expense Policy."
    }
    ],
    "messages": [{"role": "assistant", "content": 
                  
                "According to the company's policies, recognizes and pays full-time employees for the following 12 holidays plus the week between Christmas and New Year's. Here is the list of holidays for 2025 along with their calendar dates: 1. New Year’s Day - January 1, 2025, 2. Martin Luther King, Jr. Day - January 20, 2025, 3. President’s Day - February 17, 2025 4. Memorial Day - May 26, 2025, 5. Juneteenth - June 19, 2025, 6. Independence Day - July 4, 2025, 7. Labor Day - September 1, 2025, 8. Indigenous People’s Day - October 13, 2025, 9. Thanksgiving Day - November 27, 2025, 10. Day after Thanksgiving - November 28, 2025, 11. Christmas Eve - December 24, 2025, 12. Christmas Day - December 25, 2025, 13. Holiday Break (time between Christmas and New Year’s Day) - December 26, 2025 to January 1, 2026"
        }]
}

signature = infer_signature(example_input, example_output)


# COMMAND ----------


mlflow.set_registry_uri("databricks-uc")
model_name = "analytics.models.chatbot_agent"




with mlflow.start_run(run_name="chatbot_hr",nested=True) as run:
    
    model_info = mlflow.langchain.log_model(
    lc_model="./graph.py",
    artifact_path="langraph",
    registered_model_name=model_name,
    pip_requirements=[
        "mlflow==" + mlflow.__version__,
        "langchain==" + langchain.__version__,
        "databricks-vectorsearch",
        "langchain-community=="+langchain_community.__version__,
        "langgraph==0.0.51",
        "langchain-core=="+langchain_core.__version__,
        "dotenv",
        "requests",
        "datetime"
          
         ],

    signature=signature,
   
    )
    model_uri = model_info.model_uri

mlflow.end_run()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test published model with Model URI

# COMMAND ----------

## runID
#model_uri ="runs:/8ff8c93d/langraph"
#loaded_model = mlflow.langchain.load_model(model_uri)

## Run ID from the experiment tracking in mlflow Experiments - /ChatWithMyDocumentsDeployment/4_HR_Chatbot_Deploy_Agent_JIRA_1619

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_uri ="models:/analytics.models.chatbot_agent/29"
loaded_model = mlflow.langchain.load_model(model_uri)

# COMMAND ----------

from IPython.display import Image, display

display(Image(loaded_model.get_graph().draw_mermaid_png()))

# COMMAND ----------

loaded_model

# COMMAND ----------

loaded_model.nodes

# COMMAND ----------

# Show inference and message history functionality
print("-------- Message 1 -----------")
payload  ={"messages": [{"role": "user", "content": "you are not polite"}]} 

response = loaded_model.invoke(payload)



# COMMAND ----------


from langgraph_utils import get_most_recent_message
from langgraph_utils import _langgraph_message_to_mlflow_message


# COMMAND ----------

_langgraph_message_to_mlflow_message(response.get("messages")[-1])

# COMMAND ----------


print(f"User: {example_input}")
print(f"Agent: {get_most_recent_message(response)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Published model with Langgraph Wrapper
# MAGIC

# COMMAND ----------


from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import Runnable

# COMMAND ----------



import uuid
from typing import Any, Dict, Optional
from langchain_core.runnables import Runnable
from langchain_core.messages import AIMessage, HumanMessage



class LanggraphRunnable(Runnable):

    def invoke(self, input_text: str, config: Optional[Dict] = None) -> str:
        run_config = config or {}
        
        # Define the input payload for the graph
        input_payload = {'messages': [HumanMessage(content=input_text)]}

        final_state = loaded_model.invoke(input_payload, config=run_config)
        
        # Safely extract the last message from the final state dictionary
        if final_state and "messages" in final_state:
            if final_state["messages"]:
                last_message = final_state["messages"][-1]
                if isinstance(last_message, AIMessage):
                    return last_message.content

        # This will be returned only if something goes wrong
        return "Error: Could not extract a valid response from the chatbot."


langraph_runnable = LanggraphRunnable()



# COMMAND ----------

question = "What are the list of holidays for 2025?"

current_conversation_id = str(uuid.uuid4())
#print(f"Starting new conversation with Thread ID: {current_conversation_id}")
# Create the 'config' dictionary in the required format:
run_config = {"configurable": {"thread_id": current_conversation_id}}



# Call invoke with the question AND the correctly structured 'run_config':
answer = langraph_runnable.invoke(question, config=run_config)


if answer:
    print("Chatbot Response:", answer)
else:
    print("No response from the chatbot.")


# COMMAND ----------


question = "now provide me answer with the actual calendar dates 2025"
# Call invoke with the question AND the correctly structured 'run_config':
answer = langraph_runnable.invoke(question, config=run_config)


if answer:
    print("Chatbot Response:", answer)
else:
    print("No response from the chatbot.")

# COMMAND ----------

question = "How do I do expenses when I travel?"
# Call invoke with the question AND the correctly structured 'run_config':
answer = langraph_runnable.invoke(question, config=run_config)


if answer:
    print("Chatbot Response:", answer)
else:
    print("No response from the chatbot.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test published model with Serving endpoint

# COMMAND ----------

example_input = {
    "messages": [
        {"role": "user", "content": "What are the key takeaways of harassment policy?"}
    ]
}

validation_result = mlflow.models.validate_serving_input(
    model_uri=model_uri,
    serving_input=example_input
)
print(f"Validation Result: {validation_result}")

# COMMAND ----------



import os
import requests


 
# Databricks API Config
DATABRICKS_ENDPOINT = "https://adb-207441.net/serving-endpoints/hr_chatbot_agent/invocations"



# Request Payload
payload  ={"messages": [{"role": "user", "content": "How do I expenses when I travel?"}]} 

# Headers
headers = {
    "Authorization": f"Bearer {os.environ['DATABRICKS_TOKEN']}",
    "Content-Type": "application/json"
}
response = requests.post(DATABRICKS_ENDPOINT, headers=headers, json=payload)
result = response.get("messages")[-1] if response.status_code == 200 else {"error": response}
result
