import os 
from dotenv import load_dotenv
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers.json import SimpleJsonOutputParser


_ = load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

client = OpenAI()
chatClient = ChatOpenAI(model='gpt-4o-mini') 

json_prompt = PromptTemplate.from_template(
    'Return a JSON object with an `answer` key that answers the following question: {question}'
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | client | json_parser

res = json_chain.invoke(
    input='What is the capital of France?'
)
print(f'Response: {res}')
print('\n----------\n')


print(f'Output Parser Instruction: {json_parser.get_format_instructions()}')
print('\n----------\n')


### Custom Output Parser using pydantic

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# define a pydantic obj with the desired output format.
class Joke(BaseModel):
    setup: str = Field(description='question to set up a joke')
    punchline: str = Field(description='answer to resolve the joke')



# define the parser referring the pydantic object
parser = JsonOutputParser(pydantic_object=Joke)
prompt = PromptTemplate(
    template='Answer the user query. \n {format_instructions} \n {query}',
    input_variables=['query'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

chain = prompt | chatClient | parser 
res = chain.invoke(
    input='Tell me a joke about a cat.'
)

print(f'Parsed Response: {res}')
print('\n----------\n')

