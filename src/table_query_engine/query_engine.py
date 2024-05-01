import os
import pandas as pd
from .models import QueryResponse

from llama_index.core.query_engine import NLSQLTableQueryEngine
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)
from llama_index.core import PromptTemplate
from IPython.display import Markdown, display
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.query_engine import NLSQLTableQueryEngine

template_dict = {"B" :  """<s>[INST] <<SYS>>\nYou are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>\n\n
You are a SQLite expert. Given an input question,create a syntactically correct SQLite query to run.
You can order the results to return the most informative data in the database.
You are working with only one table named financial_statement.

DO WRAP EVERY COLUMN NAME WITH (").
For example: DO: "Market Cap".
DO NOT: Market Cap
Use the following format:

SQLQuery: SQL Query to run

Only use the following columns of the financial_statement table:
Year
Description: Year of financial statement

Data Type: Categorical;

Company
ชื่อบริษัท
Data Type: Text;

Category
หมวดหมู่
Data Type: Categorical;

Market Cap(in B USD)
มูลค่าตลาด (พันล้าน USD)

Revenue

Gross Profit
ผลกำไรขั้นต้น

Net Income
ผลกำไรสุทธิ

Earning Per Share
กำไรต่อหุ้น

EBITDA:
กำไรก่อนดอกเบี้ย ภาษี ค่าเสื่อมราคาและค่าตัดจำหน่าย

Share Holder Equity
ส่วนของผู้ถือหุ้น

Cash Flow from Operating

Cash Flow from Investing
Description: Any inflows or outflows of cash from a company's long-term investments.
กระแสเงินสดจากการลงทุน

Cash Flow from Financial Activities
กระแสเงินสดจากกิจกรรมจัดหาเงิน

Current Ratio
Data Type: Date;

Debt/Equity Ratio
อัตราส่วน หนี้สิน ต่อ ทุน

ROE
ผลตอบแทนต่อส่วนของผู้ถือหุ้น

ROA
อัตราผลตอบแทนจากสินทรัพย์ทั้งหมด

ROI
ผลตอบแทนจากการลงทุนนั่นเอง.

Net Profit Margin
อัตรากำไร (%)

Free Cash Flow per Share


Return on Tangible Equity


Number of Employees
Description: Number of Employee in each company
จำนวนพนักงาน

Inflation Rate(in US)
Description: the rate of increase in prices over a given period of time in US
อัตราเงินเฟ้อ

Here's some example rows from the table:
<Table>
Year	Company	Category	Market Cap(in B USD)	Revenue	Gross Profit	Net Income	Earning Per Share	EBITDA	Share Holder Equity	Cash Flow from Operating	Cash Flow from Investing	Cash Flow from Financial Activities	Current Ratio	Debt/Equity Ratio	ROE	ROA	ROI	Net Profit Margin	Free Cash Flow per Share	Return on Tangible Equity	Number of Employees	Inflation Rate(in US)
MSFT	IT	238.78	62484	50089	18760	2.1	26771	46175	24073	-11314	-13291	2.1293	0.1286	40.628	21.7853	36.7023	30.0237	0.7057	57.5054	94000	1.64
</Table>
###RULES
Remember to DO WRAP EVERY COLUMN NAME WITH single quote(").
For example: DO: "Market Cap".
DO NOT: Market Cap
And DO NOT CHANGE Company symbol to it's full name
for example: MSFT should stay as MSFT. NEVER Microsoft
IF the question is not related to the columns or table. Just say I don't know.
This is some relevant question and their SQL examples:
Question:บริษัท MSFT จัดอยู่ใน Category อะไร ในปี 2022
SQL Query:SELECT Category FROM financial_statement WHERE Year = 2022 AND Company = "MSFT" 
Question:ผลรวมกำไรขั้นต้นของบริษัท XXX 5 ปีย้อนหลัง (ตั้งแต่ปี 2019-2023) มากกว่า ผลรวมกำไรขั้นต้นของบริษัท YYY 5 ปีย้อนหลัง (ตั้งแต่ปี 2019-2023) หรือไม่ กรุณาตอบเป็น 'True' หรือ 'False',
SQL Query:SELECT SUM(CASE WHEN a.Company = 'XXX' THEN a."Gross Profit" ELSE 0 END) AS XXX_Gross_Profit, SUM(CASE WHEN a.Company = 'YYY' THEN a."Gross Profit" ELSE 0 END) AS YYY_Gross_Profit FROM (SELECT * FROM financial_statement WHERE Company IN ('XXX', 'YYY') AND Year BETWEEN 2019 AND 2023) a;
Question: {query_str}
SQL Query: 
[/INST]"""
 , 
"A" : """<<SYS>>\nYou are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>\n\n
You are a SQLite expert. Given an input question,create a syntactically correct SQLite query to run.
You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the table below. Be careful to not query for columns that do not exist.
You are working with only one table named online_shopping.
Unless the query specifies the number of results, you MUST give only 1 answer, using LIMIT 1
DO WRAP EVERY COLUMN NAME WITH (").
For example: DO: "Market Cap".
DO NOT: Market Cap
Use the following format:

SQLQuery: SQL Query to run


Only use the following columns of the online_shopping table:
### CustomerID
รหัสลูกค้า

### Gender
Description: Gender of the customer (e.g., M,F).
เพศ

### Location
สถานที่ เมือง
Data Type: Text;
### Tenure_Months

### Transaction_ID

### Transaction_Date
วันที่ทำธุรกรรม

### Product_SKU
รหัสสินค้า

### Product_Description

### Product_Category:
ประเภทสินค้า

### Quantity
จำนวนสินค้าที่ซื้อ

### Avg_Price
ราคาเฉลี่ย

### Total_Price
ราคารวม ไม่รวมค่าจัดส่ง

### Delivery_Charges
ค่าจัดส่ง

### Date
วันที่ YYYY-MM-DD

### Month
เดือน

Here's some example rows from the table:
<Table>
CustomerID	Gender	Location	Tenure_Months	Transaction_ID	Transaction_Date	Product_SKU	Product_Description	Product_Category	Quantity	Avg_Price	Total_Price	Delivery_Charges	Date	Month
17850	M	Chicago	12	16679	2019-01-01	GGOENEBJ079499	Nest Learning Thermostat 3rd Gen-USA - Stainless Steel	Nest-USA	1	153.71	153.71	6.5	1/1/2019	1
17850	M	Chicago	12	16680	2019-01-01	GGOENEBJ079499	Nest Learning Thermostat 3rd Gen-USA - Stainless Steel	Nest-USA	1	153.71	153.71	6.5	1/1/2019	1
17850	M	Chicago	12	16696	2019-01-01	GGOENEBQ078999	Nest Cam Outdoor Security Camera - USA	Nest-USA	2	122.77	245.54	6.5	1/1/2019	1
17850	M	Chicago	12	16699	2019-01-01	GGOENEBQ079099	Nest Protect Smoke + CO White Battery Alarm-USA	Nest-USA	1	81.5	81.5	6.5	1/1/2019	1
17850	M	Chicago	12	16700	2019-01-01	GGOENEBJ079499	Nest Learning Thermostat 3rd Gen-USA - Stainless Steel	Nest-USA	1	153.71	153.71	6.5	1/1/2019	1
</Table>

###RULES
Remember to DO WRAP EVERY COLUMN NAME WITH single quote(").
For example: DO: "Market Cap".
DO NOT: Market Cap

IF the question is not related to the columns or table. Just say I don't know.
Here's some example of relevant question and queries:
Question: ผู้หญิงหรือผู้ชายใครมียอดจำนวนเงินสั่งซื้อไม่รวมค่าขนส่งมากกว่ากัน กรุณาตอบแค่ `ผู้หญิง` หรือ `ผู้ชาย`
SQL Query: SELECT   CASE     WHEN SUM(CASE WHEN Gender = 'M' THEN Total_Price ELSE 0 END) > SUM(CASE WHEN Gender = 'F' THEN Total_Price ELSE 0 END) THEN 'ผู้ชาย'    ELSE 'ผู้หญิง'  END AS Answer FROM   online_shopping;
Question: {query_str}
SQL Query:[/INST]
"""}

class QueryEngine:
    def __init__(self, llm, df):
        self.query_engine = NLSQLTableQueryEngine(sql_database=df, llm=llm, verbose=True)

    def __call__(self, query_str) -> QueryResponse:
        response = self.query_engine.query(query_str)
        return QueryResponse(response=response.response)


def initialize_query_engine(table : str):
    
    llm = HuggingFaceLLM(
    model_name="/project/lt900052-ai2414/Gital/SuperAI_LLM_FineTune/new_checkpoint_3",
    tokenizer_name="/project/lt900052-ai2414/Gital/SuperAI_LLM_FineTune/new_checkpoint_3",
    device_map="auto",
)

    Settings.embed_model = None

    engine = create_engine("sqlite:////scratch/lt900052-ai2414/exp.db")

    sql_database = SQLDatabase(engine, include_tables=["online_shopping",'financial_statement'])

    query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database, tables=["online_shopping"], llm=llm,
        verbose = True,
    )

    new_prompt_str = template_dict[table]

    resp_synthesis_prompt_str = ("""
[INST]<<SYS>>
You are a question answering assistant. Answer the question as truthful and helpful as possible คุณคือผู้ช่วยตอบคำถาม จงตอบคำถามอย่างถูกต้องและมีประโยชน์ที่สุด<</SYS>>
Pay attention to the format the user wants the answer in.
Be careful about decimal places specified by the user. 
Pay attention to the format the user wants the answer in. If the user wants JSON, reply in JSON. If the uswe wants array, reply in array
You MUST CAREFULLY FOLLOW ANY INSTRUCTION IN THE QUESTION.
For example: ทศนิยม 2 ตำแหน่ง means 2 decimal places -> 1.00
Do not give any reasoning to your answer.
To answer the question, I have already executed SQL query to find the answer for you. You need to use the provided context to answer the original question.

Context information is below.\n"
    "---------------------\n"
    {context_str}
    "---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "Query: {query_str} \n"
    "Answer:[/INST]"
""")

    new_prompt = PromptTemplate(new_prompt_str)
    resp_synthesis_prompt = PromptTemplate(resp_synthesis_prompt_str)
    query_engine.update_prompts({
        "sql_retriever:text_to_sql_prompt": new_prompt,
        "response_synthesis_prompt": resp_synthesis_prompt
    })
    return query_engine
