import ollama
from llm_axe import OnlineAgent, OllamaChat

model_name = "deepseek-r1:1.5b"
llm = OllamaChat(model=model_name)
messages = [{"role": "system", "content": "hello what can I do?"},
            {"role": "user", "content": "hello"}]

response = ollama.chat(model=model_name, messages=messages)
bot_reply = response["message"]["content"]
print("bot:", bot_reply)

online_agent = OnlineAgent(llm=llm)
messages.append({"role": "assistant", "content": bot_reply})

while True:
    user_input = input("you: ")
    if not user_input:
        break

    if "search" in user_input.lower():
        query = f"Find reliable information about {user_input} from trusted sources."
        search_result = online_agent.search(query, max_results=10, relevance_threshold=0.8)
        summary_prompt = f"Summarize this search result in simple words: {search_result}"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": summary_prompt}])
        bot_reply = response['message']['content']
        print("bot:", bot_reply)
    else:
        messages.append({"role": "user", "content": user_input})
        response = ollama.chat(model=model_name, messages=messages)
        bot_reply = response["message"]["content"]
        print("bot:", bot_reply)
        messages.append({"role": "assistant", "content": bot_reply})