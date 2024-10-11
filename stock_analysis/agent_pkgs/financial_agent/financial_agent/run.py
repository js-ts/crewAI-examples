import os 
import requests 
import html2text 
import re 
from crewai.agent import Agent 
from crewai.project.annotations import agent 
from crewai_tools.tools.base_tool import BaseTool 
from typing import Any 
from typing import Optional 
from typing import Type 
from pydantic.v1.main import BaseModel 
from crewai_tools.tools.rag.rag_tool import RagTool 
from sec_api.index import QueryApi 
from embedchain.models.data_type import DataType 
from pydantic.v1.fields import Field 
from crewai_tools.tools.website_search.website_search_tool import WebsiteSearchTool 
from crewai_tools.tools.scrape_website_tool.scrape_website_tool import ScrapeWebsiteTool 
from crewai import Task
from dotenv import load_dotenv
from financial_agent.schemas import InputSchema
from naptha_sdk.utils import get_logger

logger = get_logger(__name__)

load_dotenv()

class CalculatorTool(BaseTool):
    name: str = "Calculator tool"
    description: str = (
        "Useful to perform any mathematical calculations, like sum, minus, multiplication, division, etc. The input to this tool should be a mathematical  expression, a couple examples are `200*7` or `5000/2*10."
    )

    def _run(self, operation: str) -> int:
        # Implementation goes here
        return eval(operation)

class FixedSEC10KToolSchema(BaseModel):
    """Input for SEC10KTool."""
    search_query: str = Field(
        ...,
        description="Mandatory query you would like to search from the 10-K report",
    )

class SEC10KToolSchema(FixedSEC10KToolSchema):
    """Input for SEC10KTool."""
    stock_name: str = Field(
        ..., description="Mandatory valid stock name you would like to search"
    )

class SEC10KTool(RagTool):
    name: str = "Search in the specified 10-K form"
    description: str = "A tool that can be used to semantic search a query from a 10-K form for a specified company."
    args_schema: Type[BaseModel] = SEC10KToolSchema

    def __init__(self, stock_name: Optional[str] = None, **kwargs):
        print("enter init")
        # exit()
        super().__init__(**kwargs)
        if stock_name is not None:
            content = self.get_10k_url_content(stock_name)
            if content:
                self.add(content)
                # print("exit init")
                # exit()
                self.description = f"A tool that can be used to semantic search a query from {stock_name}'s latest 10-K SEC form's content as a txt file."
                self.args_schema = FixedSEC10KToolSchema
                self._generate_description()

    def get_10k_url_content(self, stock_name: str) -> Optional[str]:
        """Fetches the URL content as txt of the latest 10-K form for the given stock name."""
        try:
            queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{stock_name} AND formType:\"10-K\""
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{ "filedAt": { "order": "desc" }}]
            }
            filings = queryApi.get_filings(query)['filings']
            if len(filings) == 0:
                print("No filings found for this stock.")
                return None

            url = filings[0]['linkToFilingDetails']
            
            headers = {
                "User-Agent": "crewai.com bisan@crewai.com",
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  
            h = html2text.HTML2Text()
            h.ignore_links = False
            text = h.handle(response.content.decode("utf-8"))

            text = re.sub(r"[^a-zA-Z$0-9\s\n]", "", text)
            return text
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error fetching 10-K URL: {e}")
            return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        kwargs["data_type"] = DataType.TEXT
        super().add(*args, **kwargs)

    def _run(self, search_query: str, **kwargs: Any) -> Any:
        return super()._run(query=search_query, **kwargs)

class FixedSEC10QToolSchema(BaseModel):
    """Input for SEC10QTool."""
    search_query: str = Field(
        ...,
        description="Mandatory query you would like to search from the 10-Q report",
    )

class SEC10QToolSchema(FixedSEC10QToolSchema):
    """Input for SEC10QTool."""
    stock_name: str = Field(
        ..., description="Mandatory valid stock name you would like to search"
    )

class SEC10QTool(RagTool):
    name: str = "Search in the specified 10-Q form"
    description: str = "A tool that can be used to semantic search a query from a 10-Q form for a specified company."
    args_schema: Type[BaseModel] = SEC10QToolSchema

    def __init__(self, stock_name: Optional[str] = None, **kwargs):
        print("enter init")
        # exit()
        super().__init__(**kwargs)
        if stock_name is not None:
            content = self.get_10q_url_content(stock_name)
            if content:
                self.add(content)
                self.description = f"A tool that can be used to semantic search a query from {stock_name}'s latest 10-Q SEC form's content as a txt file."
                self.args_schema = FixedSEC10QToolSchema
                self._generate_description()

    def get_10q_url_content(self, stock_name: str) -> Optional[str]:
        """Fetches the URL content as txt of the latest 10-Q form for the given stock name."""
        try:
            queryApi = QueryApi(api_key=os.environ['SEC_API_API_KEY'])
            query = {
                "query": {
                    "query_string": {
                        "query": f"ticker:{stock_name} AND formType:\"10-Q\""
                    }
                },
                "from": "0",
                "size": "1",
                "sort": [{ "filedAt": { "order": "desc" }}]
            }
            filings = queryApi.get_filings(query)['filings']
            if len(filings) == 0:
                print("No filings found for this stock.")
                return None

            url = filings[0]['linkToFilingDetails']
            
            headers = {
                "User-Agent": "crewai.com bisan@crewai.com",
                "Accept-Encoding": "gzip, deflate",
                "Host": "www.sec.gov"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            h = html2text.HTML2Text()
            h.ignore_links = False
            text = h.handle(response.content.decode("utf-8"))

            # Removing all non-English words, dollar signs, numbers, and newlines from text
            text = re.sub(r"[^a-zA-Z$0-9\s\n]", "", text)
            return text
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error fetching 10-Q URL: {e}")
            return None

    def add(self, *args: Any, **kwargs: Any) -> None:
        kwargs["data_type"] = DataType.TEXT
        super().add(*args, **kwargs)

    def _run(self, search_query: str, **kwargs: Any) -> Any:
        return super()._run(query=search_query, **kwargs)

agents_config = {'financial_analyst': {'role': 'The Best Financial Analyst\n', 'goal': 'Impress all customers with your financial data and market trends analysis\n', 'backstory': 'The most seasoned financial analyst with lots of expertise in stock market analysis and investment strategies that is working for a super important customer.\n'}, 'research_analyst': {'role': 'Staff Research Analyst\n', 'goal': 'Being the best at gathering, interpreting data and amazing your customer with it\n', 'backstory': "Known as the BEST research analyst, you're skilled in sifting through news, company announcements, and market sentiments. Now you're working on a super important customer.\n"}, 'investment_advisor': {'role': 'Private Investment Advisor\n', 'goal': 'Impress your customers with full analyses over stocks and complete investment recommendations\n', 'backstory': "You're the most experienced investment advisor and you combine various analytical insights to formulate strategic investment advice. You are now working for a super important customer you need to impress.\n"}}

def financial_agent() -> Agent:
    return Agent(
        config=agents_config['financial_analyst'],
        verbose=True,
        tools=[
            ScrapeWebsiteTool(),
            WebsiteSearchTool(),
            CalculatorTool(),
            SEC10QTool("AMZN"),
            SEC10KTool("AMZN"),
        ]
    )

def run(inputs: InputSchema, *args, **kwargs):
    financial_agent_0 = financial_agent()

    tool_input_class = globals().get(inputs.tool_input_type)
    tool_input = tool_input_class(**inputs.tool_input_value)
    method = getattr(financial_agent_0, inputs.tool_name, None)

    return method(tool_input)

if __name__ == "__main__":
    from naptha_sdk.utils import load_yaml
    from financial_agent.schemas import InputSchema

    cfg_path = "financial_agent/component.yaml"
    cfg = load_yaml(cfg_path)

    # You will likely need to change the inputs dict
    inputs = {"tool_name": "execute_task", "tool_input_type": "Task", "tool_input_value": {"description": "What is the market cap of AMZN?", "expected_output": "The market cap of AMZN"}}
    inputs = InputSchema(**inputs)

    response = run(inputs)
    print(response)
