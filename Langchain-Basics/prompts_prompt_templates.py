import os 
from dotenv import load_dotenv, find_dotenv 
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get('OPENAI_API_KEY') 
os.environ['OPENAI_API_KEY'] = openai_api_key 


############################# Completion Model(latest) #########################
print('Working with completion version:')
print('\n----------\n')
client = OpenAI() 


prompt_template = PromptTemplate.from_template('Tell me a {adjective} story about {topics}.')
llmModelPrompt = prompt_template.format(adjective='curious', topics='the universe')

res = client.invoke(llmModelPrompt)

print(f'Response: {res}')

print('\n----------\n')

############################# Chat Model #########################
print('Working with chat version:')
print('\n----------\n')
chatClient = ChatOpenAI(model='gpt-4o-mini')

chat_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are an {profession} expert in the field of {field}.'),
        ('human', 'Hello, mr. {profession} expert. Can you tell me one fun fact?'),
        ('ai', 'sure'),
        ('human', '{user_input}')
    ]
)

llmChatPrompt = chat_template.format(
    profession='astrophysicist',
    field='universe',
    user_input='Tell me one fun fact about the universe'
)

res = chatClient.invoke(llmChatPrompt)

print(f'Response: {res.content}')

print('\n----------\n')

