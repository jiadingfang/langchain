from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI

agent_executor = create_python_agent(
    llm=ChatOpenAI(temperature=0, model="gpt-4-0613"),
    tool=PythonREPLTool(),
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

# refer questions
prompt = '''scene0536_01 has objects in it and I'll give you some quantitative descriptions. All descriptions are in right-handed Cartesian coordinate system with x-y-z axes, where x represents left-right, y represents forward-backward, and z represents up-down. Objects are:
A table with id 20, its center position is [6.5921674, 2.8146675, 0.5843334], and its size in x,y,z direction is [0.50327635, 0.4665177, 0.61986494].
A table with id 19, its center position is [3.6230552, 2.7831311, 0.4653807], and its size in x,y,z direction is [0.4619925, 0.5032735, 0.60380846].
A stairs with id 8, its center position is [8.354945, 2.8620052, 0.7037763], and its size in x,y,z direction is [0.45843983, 0.5531683, 0.53641963].
A pillow with id 9, its center position is [4.9617434, 3.0305147, 0.87839276], and its size in x,y,z direction is [0.6176739, 0.26221752, 0.55044407].
Find the referred object in the following sentence and display its id only:
select the table that is in the center of the stairs and the pillow.
You should work this out in a step by step way to make sure we have the right answer.'''

agent_executor.run(prompt)




