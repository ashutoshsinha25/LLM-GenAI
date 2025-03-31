import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq 


_ = load_dotenv()

os.environ['GROQ_API_KEY'] = os.environ.get('GROQ_API_KEY')
# print(os.environ['GROQ_API_KEY'])



print('Working with chat version:')
 
llamaChatModel = ChatGroq(
    model='llama3-70b-8192'
)

mistralChatModel = ChatGroq(
    model='mistral-saba-24b'
)

messages = [
    ('system', 'You are an astrophysicist, expert in the field of universe.'),
    ('human', 'Tell me one fun fact about the universe')
]

print('\n----------\n')

print("""Llama Chat Model:""")
res = llamaChatModel.invoke(messages)
print(f'Response: {res.content}')

print('\n----------\n')

print("""Mistral Chat Model:""")
res = mistralChatModel.invoke(messages)
print(f'Response: {res.content}')

print('\n----------\n')

