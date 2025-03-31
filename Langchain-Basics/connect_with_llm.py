'''
- langchain divides LLMs into two types:
    -LLM Model : text completion model (old way)
    -Chat Model: Conversation model

'''

######################### Completion Model(latest) #########################
import os 
from dotenv import load_dotenv
from langchain_openai import OpenAI
_ = load_dotenv() 

openai_key = os.environ.get("OPENAI_API_KEY")
os.environ['OPENAI_API_KEY'] = openai_key

# print(os.environ['OPENAI_API_KEY']
client = OpenAI()

print('Working with completion version:')
print('\n----------\n')

res = client.invoke(
    'Tell me one fun fact about the universe',
)


print(f'Response: {res}')

print('\n----------\n')

print('Streaming completion:')

for chunk in client.stream('Tell me one fun fact about the universe'):
    print(chunk, end='', flush=True)


print('\n----------\n')



## Temperature: more or less creativity 

creativeClient = OpenAI(temperature = 0.9) # high creative 

res = creativeClient.invoke('Tell me about the universe')
print(f'Response: {res}')

print('\n----------\n')



    
######################### Chat Model #########################
print('Working with chat version:')
from langchain_openai import ChatOpenAI 
import time 
# chatClient = ChatOpenAI(model='gpt-3.5-turbo-0125')
chatClient = ChatOpenAI(model='gpt-4o-mini')

# this part is different from completion model
message = [('system', 'you are an expert in Universe.'), # system-> role
            ('human', 'Tell me one fun fact about the universe')] # human-> qestion/prompt 
            

res = chatClient.invoke(message)
print(f'Response: {res.content}')

# res.response_metadata 
# res.schema() 

print('\n----------\n')


print('Streaming chat:')
for chunk in chatClient.stream(message):
    # add time delay to simulate streaming
    time.sleep(0.3)
    print(chunk, end='', flush=True)

print('\n----------\n')



######################### Chat Model Older Way #########################
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate 

messages = [
    SystemMessage(content='you are an expert in Universe.'),
    HumanMessage(content='Tell me one fun fact about the universe')
]

res = chatClient.invoke(messages)
print(f'Response: {res.content}')

print('\n----------\n')

print('Streaming chat:')
for chunk in chatClient.stream(messages):
    # add time delay to simulate streaming
    time.sleep(0.3)
    print(chunk.content, end='', flush=True)

print('\n----------\n')



######################### Another way to do similar thing #########################
print('Working with older chat version: ')
prompt = ChatPromptTemplate.from_messages([
    ('system', 'you are an expert in Universe.'),
    ('human', 'Tell me one fun fact about the universe')
])

chain = prompt | chatClient
res = chain.invoke({
    'profession' : 'astro physicist',
    'topic': 'universe',
    'input': 'Tell me one fun fact about the universe'
})
print(f'Response: {res.content}')

print('\n----------\n')