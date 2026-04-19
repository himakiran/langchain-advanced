from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

# Define a function to get the OllamaLLM model with the specified parameters

def get_model(model_name="gemma4:31b", temperature=0.8, top_p=0.95, top_k=40, num_predict=128):
    model = OllamaLLM(
        model=model_name,
        temperature=temperature, # Higher values (e.g., 0.8) make the output more creative, while lower values (e.g., 0.2) make it more deterministic. default is 0.8
        top_p=top_p, # Used along with top_k. A higher value (e.g., 0.95) allows for more diverse outputs by sampling from a larger pool of tokens, while a lower value (e.g., 0.5) makes the output more focused and deterministic. default is 0.95
        top_k=top_k, #Reduces probability of generating nonsense A higher value (e.g., 40) allows for more diverse outputs by sampling from a larger pool of tokens, while a lower value (e.g., 16) makes the output more focused and deterministic. default is 40
        num_predict=num_predict, # The maximum number of tokens to generate in the response. A higher value (e.g., 128) allows for longer responses, while a lower value (e.g., 512) limits the response length. default is 128
        # base_url="http://localhost:11434",
        # other params...
    )
    return model

def play_with_params():
    model_names = {1: "deepseek-r1:32b", 2: "qwen3.6:latest", 3: "huihui_ai/gemma-4-abliterated:e4b", 4: "huihui_ai/gemma-4-abliterated:e4b-old", 5: "minimax-m2.7:cloud", 6: "gemma4:31b", 7: "gemma4:latest", 8: "nomic-embed-text:latest", 9: "gemma3n:latest", 10: "gemma3:27b", 11: "qwen3.5:9b", 12: "llama3.1:latest"}
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    top_ps = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    top_ks = [16, 32, 40, 64, 128]
    num_predicts = [64, 128, 256, 512,1024,2048]
    try:
        model_name = int(input(f"Choose a model from the following list: {model_names}\n"))
        if model_name not in model_names:
            print("Invalid model selection. Defaulting to gemma4:31b.")
    except ValueError:
            model_name = 6 # default to gemma4:31b if invalid input
    try:
        temperature = float(input(f"Choose a temperature from the following list - Higher is more creative: {temperatures}\n"))
        if temperature not in temperatures:
            print("Invalid temperature selection. Defaulting to 0.8.")
    except ValueError:
        temperature = 0.8 # default to 0.8 if invalid input
    try:
        top_p = float(input(f"Choose a top_p from the following list - Higher is more diverse: {top_ps}\n"))
        if top_p not in top_ps:
            print("Invalid top_p selection. Defaulting to 0.9.")
    except ValueError:
            top_p = 0.9 # default to 0.9 if invalid input
    try:
        top_k = int(input(f"Choose a top_k from the following list - Higher is more diverse: {top_ks}\n"))
        if top_k not in top_ks:
            print("Invalid top_k selection. Defaulting to 32.")
    except ValueError:
            top_k = 32 # default to 32 if invalid input
    try:
        num_predict = int(input(f"Choose a num_predict from the following list - Higher is longer: {num_predicts}\n"))
        if num_predict not in num_predicts:
            print("Invalid num_predict selection. Defaulting to 2048.")
    except ValueError:
            num_predict = 2048 # default to 2048 if invalid input
    
    return model_names[model_name], temperature, top_p, top_k, num_predict

def get_prompt_string():
    optimist_prompt_string = """You are a diehard optimist. You see the best in everything. You will always find the silver lining in every cloud. You will find a positive spin on every situation. Whatever is asked you shall always find a way to see the good in it and answer positively. Reply to the following query as an optimist in line with your persona: {user_query}"""
    pessimist_prompt_string = """You are a diehard pessimist. You see the worst in everything. You will always find the dark cloud in every silver lining. You will find a negative spin on every situation. Whatever is asked you shall always find a way to see the bad in it and answer negatively. Reply to the following query as a pessimist in line with your persona: {user_query}"""
    realist_prompt_string = """You are a realist. You see things as they are. You will always find the truth in every situation. You will find a balanced spin on every situation. Whatever is asked you shall always find a way to see the reality in it and answer realistically. Reply to the following query as a realist in line with your persona: {user_query}"""
    conspiracy_theorist_prompt_string = """You are a conspiracy theorist. You see hidden agendas in everything. You will always find a secret plot in every situation. You will find a conspiratorial spin on every situation. Whatever is asked you shall always find a way to see the conspiracy in it and answer with a conspiracy theory. Reply to the following query as a conspiracy theorist in line with your persona: {user_query}"""
    science_fiction_writer_prompt_string = """You are a science fiction writer. You see futuristic possibilities in everything. You will always find a sci-fi twist in every situation. You will find a speculative spin on every situation. Whatever is asked you shall always find a way to see the sci-fi potential in it and answer with a science fiction story. Reply to the following query as a science fiction writer in line with your persona: {user_query}"""
    mathematician_prompt_string = """You are a mathematician. You see patterns and logic in everything. You will always find a mathematical angle in every situation. You will find a logical spin on every situation. Whatever is asked you shall always find a way to see the mathematical structure in it and answer with a mathematical explanation. Reply to the following query as a mathematician in line with your persona: {user_query}"""
    philosopher_prompt_string = """You are a philosopher. You see deep meaning in everything. You will always find a philosophical perspective in every situation. You will find an existential spin on every situation. Whatever is asked you shall always find a way to see the philosophical implications in it and answer with a philosophical reflection. Reply to the following query as a philosopher in line with your persona: {user_query}"""
    religious_fanatic_prompt_string = """You are a religious fanatic. You see divine intervention in everything. You will always find a religious explanation in every situation. You will find a spiritual spin on every situation. Whatever is asked you shall always find a way to see the religious significance in it and answer with a religious interpretation. Reply to the following query as a religious fanatic in line with your persona: {user_query}"""

    prompts_strings_dict = {
        "optimist": optimist_prompt_string,
        "pessimist": pessimist_prompt_string,
        "realist": realist_prompt_string,
        "conspiracy_theorist": conspiracy_theorist_prompt_string,
        "science_fiction_writer": science_fiction_writer_prompt_string,
        "mathematician": mathematician_prompt_string,
        "philosopher": philosopher_prompt_string,
        "religious_fanatic": religious_fanatic_prompt_string
    }
    try:        
        persona = input(f"Choose a persona from the following list: {list(prompts_strings_dict.keys())}\n")
        if persona not in prompts_strings_dict:
            print("Invalid persona selection. Defaulting to optimist.")
            persona = "optimist" # default to optimist if invalid input
    except ValueError:
        print("Invalid persona selection. Using default persona.")
        persona = "optimist" # default to optimist if invalid input
    return persona, prompts_strings_dict[persona]

# Define a function to chat with the user by getting their query, formatting it according to the selected prompting technique, and generating a response using the model's invoke method
def chat_with_user():
    model_params= play_with_params()
    model = get_model(*model_params)
    persona, get_prompt = get_prompt_string()
    prompt_template = PromptTemplate.from_template(get_prompt)

    def format_prompt(variables):
        return prompt_template.format(**variables)

    print(
        "You have selected: "
        + persona
        + "\n and the following model parameters: "
        + str(model_params)
    )

    while True:
        user_query = input("Please enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Thank you for chatting! Goodbye!")
            break

        # Refer document runnable_lambda_explanation.md for more details on RunnableLambda and how it works in this context.

        summarize_chain = (
            RunnableLambda(format_prompt) | model | StrOutputParser()
        )
        response = summarize_chain.invoke({"user_query": user_query})
        print("Response: " + response + "\n")
def main():
    chat_with_user()


if __name__ == "__main__":
    main()
