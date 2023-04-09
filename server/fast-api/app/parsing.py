import boto3
import time

from pdf2image import convert_from_path, convert_from_bytes
import io
import pandas as pd

from node import Node
from relationship import Relationship
import os
import openai
import re
import csv


# Paragraph/Special Requirments
global bp_number
bp_number = None

global extraction_method
extraction_method = None

global mk_number
mk_number = None

global turbo_ionspray
turbo_ionspray = None

global polarity
polarity = None

global drug_lower_mz
drug_lower_mz = None
global drug_upper_mz
drug_upper_mz = None

global is_lower_mz
is_lower_mz = None
global is_upper_mz
is_upper_mz = None

global lloq
lloq = None

global regression_model
regression_model = None

global calibration_range_lower
calibration_range_lower = None
global calibration_range_upper
calibration_range_upper = None
global calibration_sample_volume
calibration_sample_volume = None

global dilutent
dilutent = None
global temperature_dilutent
temperature_dilutent = None


global anticoagulant_temperature
anticoagulant_temperature = None

global special_requirments
special_requirments = None

global author
author = None



# Global Variables for Tables [44]

# Instrumentation
global g_massspec_component
global g_liquidhandling_component
g_massspec_component = None
g_liquidhandling_component = None

# Category (General)
global column_category
global column_manufacturer
column_category = None
column_manufacturer = None

# Category (Equipment)
global microbalance_manufacturer
global analytical_balance_manufacturer
global refrigerated_centrifuge_manufacturer
global pH_meter_manufacturer
global platesealer_manufacturer
microbalance_manufacturer = None
analytical_balance_manufacturer = None
refrigerated_centrifuge_manufacturer = None
pH_meter_manufacturer = None
platesealer_manufacturer = None

# Category (Pipettes)
global adjustable_pipettes_manufacturer
global pipette_tips_manufacturer
adjustable_pipettes_manufacturer = None
pipette_tips_manufacturer = None

# Category (Automation Supplies)
global reagent_troughs_provider
global automated_workstation_tips_provider
reagent_troughs_provider = None
automated_workstation_tips_provider = None

# Reference Standards
global analyte_l_parent
global analyte_l_is
global form_parent
global form_is
global molecular_weight_parent
global molecular_weight_is
global watson_id_parent
global watson_id_is
analyte_l_parent = None
analyte_l_is = None
form_parent = None
form_is = None
molecular_weight_parent = None
molecular_weight_is = None
watson_id_parent = None
watson_id_is = None

# Matrix/Species/Anticoagulant/Supplier
global matrix
global species
global anticoagulant
global supplier
matrix = None
species = None
anticoagulant = None
supplier = None

# UPLC/LC
global loop_option_settings
global elution_settings
global mobile_phase_a_settings
global mobile_phase_b_settings
loop_option_settings = None
elution_settings = None
mobile_phase_a_settings = None
mobile_phase_b_settings = None

# MS
global ion_source_settings
global ion_mode_settings
global ionization_potential_settings
global temperature_settings
global mr_pause_settings
global mr_settling_time_settings
ion_source_settings = None
ion_mode_settings = None
ionization_potential_settings = None
temperature_settings = None
mr_pause_settings = None
mr_settling_time_settings = None

# Analyte 1
global analyte_1
global peak_height_1
global retention_time_1
analyte_1 = None
peak_height_1 = None
retention_time_1 = None

# Analyte 2
global analyte_2
global peak_height_2
global retention_time_2
analyte_2 = None
peak_height_2 = None
retention_time_2 = None

# Other Important Tables
global table_standard_solution_id
global table_qc_solution_id
global table_qc_id
global table_step
global table_ions_monitored
table_standard_solution_id = None
table_qc_solution_id = None
table_qc_id = None
table_step = None
table_ions_monitored = None

OPENAI_API_KEY = ""
openai.api_key = os.getenv("OPENAI_API_KEY")



# The above is a personal API key, when deployed the API key can be changed by
# going to https://platform.openai.com/account/api-keys , logging in and generating a key from there


session = boto3.Session(
    aws_access_key_id='',
    aws_secret_access_key=''
)

os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""


searchString = text.replace("\n", " ").replace(",","")
searchString = re.sub(' +',' ', searchString)

global page1
global qc_sample_prep
global internal_standard
global procedure
global system_suitability
page1 = ''
qc_sample_prep = ''
internal_standard = ''
procedure = ''
system_suitability = ''

# Page 1
print("PAGE 1")
match = re.search(r'^.*?Page 2', searchString, re.DOTALL)

if match:
    page1 = match.group(0)
    print(page1)
    print("\nLENGTH: " + str(len(page1)))
    print("\n\n")

# (QC) Sample Preparation

print("(QC) SAMPLE PREPARATION")
match1 = re.search(r"SAMPLE PREPARATION(.*?)INTERNAL STANDARD", searchString, re.I)
qc_sample_prep = match1.group(1)
print(qc_sample_prep)
print("\nLENGTH: " + str(len(qc_sample_prep)))
print("\n\n")

# Internal Standard

print("INTERNAL STANDARD")
match2 = re.search(r"INTERNAL STANDARD(.*?)OPERATING PARAMETERS", searchString, re.I)
internal_standard = match2.group(1)
print(internal_standard)
print("\nLENGTH: " + str(len(internal_standard)))
print("\n\n")

# Procedure

print("PROCEDURE")
match3 = re.search(r"PROCEDURE(.*?)OPERATING PARAMETERS", searchString, re.I)
procedure = match3.group(1)
print(procedure)
print("\nLENGTH: " + str(len(procedure)))
print("\n\n")

# System Suitability

print("SYSTEM SUITABILITY")
match4 = re.search(r"SYSTEM SUITABILITY(.*?)SOFTWARE AND CALCULATION", searchString, re.I)
system_suitability = match4.group(1)
print(system_suitability)
print("\nLENGTH: " + str(len(system_suitability)))


#GPT Code
gpt_prompt = """ Extract the BP Number, extraction method, the species, drug, internal standard, chromatography type, interface, ion mode, MRM transition from drug, MRM transition for internal standard, LLOQ, 1/x^2 regression model, calibration range, anticoagulant, study samples, study sample temperature  given the context below?: \n
context:"""

gpt_prompt = gpt_prompt + page1


response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=gpt_prompt,
  temperature=0.0,
  max_tokens=400,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
)


print(response['choices'][0]['text'])
#print(response['choices'][0])

#prediction_table.add_data(gpt_prompt,response['choices'][0]['text'])
filename = "BP0001_para1_chatgpt_output.csv"
rows = response['choices'][0]['text'].strip().split("\n")

data_list = []
with open(filename, "w", newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  for row in rows:
    # Split row into key and value
    row_split = row.split(": ", 1)
    if len(row_split) < 2:
        key = row_split[0]
        data_list.append({key: ""})
    else:
        # If value is < 2, there was no colon in that line and
        key, value = row_split
        data_list.append({key.strip(): value})
    csvwriter.writerow([key, value])
gpt_prompt = """ Provided below in the formatting of a BP document paragraph:This analytical method is based on an automated 96-well format [Extraction Method] of drug from [species matrix]. MK-[XXXX] and stable isotope labeled internal standard ([XXX]) are chromatographed using [chromatography] and detected with tandem mass spectrometric detection employing a [turbo ionspray (TIS)] interface in the [polarity] ion mode. The Multiple Reaction Monitoring (MRM) transitions monitored were m/z [XXX ® XXX] for the drug and m/z [XXX ® XXX] for the internal standard. The lower limit of quantitation (LLOQ) for this method is [X] ng/mL with a [regression model] 1/x2 (weighting) calibration range from [X] to [XXX] ng/mL using a [X] mL [matrix] sample. Standard solutions are prepared in [diluent] and stored at [temperature°C] when not in use. [Anticoagulant is used as the anticoagulant] and [matrix] study samples are stored at [-temperature°C].
Now, with this filled BP document paragraph, extract the bracketed values given the formatting provided prior  : \n
context: \n"""

gpt_prompt = gpt_prompt + page1


response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=gpt_prompt,
  temperature=0.0,
  max_tokens=1000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
)


print(response['choices'][0]['text'])


gpt_prompt = """Extract the Diluent, Storage temperature, Solvent, Density, and an other condition if its used (typically in parenthesis) if used given the context below?: \n
\n
text: \n 
"""
gpt_prompt = gpt_prompt + qc_sample_prep

response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=gpt_prompt,
  temperature=0.0,
  max_tokens=1000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
)


print(response['choices'][0]['text'])

filename = "BP0001_para1_chatgpt_output.csv"
rows = response['choices'][0]['text'].strip().split("\n")

data_list = []
with open(filename, "a", newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  for row in rows:
    # Split row into key and value
    row_split = row.split(": ", 1)
    if len(row_split) < 2:
        key = row_split[0]
        data_list.append({key: ""})
    else:
        # If value is < 2, there was no colon in that line and
        key, value = row_split
        data_list.append({key.strip(): value})
    csvwriter.writerow([key, value])

gpt_prompt = """Given the text: 
A.	Stock QC Solution 

Weigh the compound and transfer into an amber glass vial. Dissolve in *Diluent* to make a *1.00* mg/mL SIL-*Name* stock solution* free form stock solution while correcting for potency (e.g., purity, residual solvents, and excess water content) and salt factor, as appropriate. Mix well and *sonicate x minutes (or other conditions, if used)*. Store at *storage conditions (TemperatureºC).*

<<Below is templated text for Working QC Solutions (if needed) and Matrix QC samples. >> 



B.	Working QC Solutions 

Using adjustable pipettes, transfer the volumes of each QC indicated in the table below into amber glass vials. Dilute to appropriate volume with diluent (*Diluent*). Mix well. Store at storage conditions *(TemperatureºC).*
C.	Matrix QCs 

To a polypropylene tube, add the designated spiking volume of appropriate spiking solution to blank matrix. Cap the tube and briefly vortex *followed by rotation for 15 minutes at a setting of 40 on a rotator*. Aliquot* 0.X mL* into micronic tubes, cap, and store at -70ºC. The aliquot volume and QCs volume prepared may be scaled as needed. Matrix QCs may be used from freshly prepared or frozen samples.  

Use the text surrounded by the ** to extract the values from the following text with a similar structure: \n text: \n


"""

gpt_prompt = gpt_prompt + internal_standard

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=gpt_prompt,
    temperature=0.0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

print(response['choices'][0]['text'])

filename = "BP0001_para1_chatgpt_output.csv"
rows = response['choices'][0]['text'].strip().split("\n")

data_list = []
with open(filename, "a", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in rows:
        # Split row into key and value
        row_split = row.split(": ", 1)
        if len(row_split) < 2:
            key = row_split[0]
            data_list.append({key: ""})
        else:
            # If value is < 2, there was no colon in that line and
            key, value = row_split
            data_list.append({key.strip(): value})
        csvwriter.writerow([key, value])


gpt_prompt = """ Use the text surrounded by the ** to extract the values from the following text with a similar structure.


The glassware, reagents, and dilutions described in this procedure should serve as a guide and may be substituted for their equivalent when necessary to obtain similar results. 

Blank *matrix*: 

Thaw and *centrifuge for 5 minutes at 2800 rpm* 

*Stabilize blank matrix daily (if applicable)* 

Describe storage conditions of treated matrix e.g., store on wet ice until ready to use 



Biological *matrix* samples (e.g., unknown matrix samples, matrix QCs): 

Allow samples to thaw completely 

*Describe vortexing, rotating and centrifuge conditions <<Vortex samples for x minute at setting x>> <<Rotate sample rack on lab rotator for 10 minutes at setting 40>> <<Centrifuge for 5 minutes at 4000 rpm>> *


text: \n
"""
gpt_prompt = gpt_prompt + procedure

response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=gpt_prompt,
    temperature=0.0,
    max_tokens=1000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

print(response['choices'][0]['text'])

filename = "BP0001_para1_chatgpt_output.csv"
rows = response['choices'][0]['text'].strip().split("\n")

# Split each row into a list of columns using the colon separator
data_list = []
with open(filename, "a", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    for row in rows:
        # Split row into key and value
        row_split = row.split(": ", 1)
        if len(row_split) < 2:
            key = row_split[0]
            data_list.append({key: ""})
        else:
            # If value is < 2, there was no colon in that line and
            key, value = row_split
            data_list.append({key.strip(): value})
        csvwriter.writerow([key, value])


gpt_prompt = """ 
Use the text surrounded by the ** to extract the values from the following text with a similar structure. When you extract can you output as a name:value format please. and for the describe proceducure store it as procedure: all the bulletpoints that correlate. MK-XXXX is MK Number. Neat solution is the ng/ml.

When assaying biological extracts: 

An extracted system suitability sample at the LLOQ will be injected prior to sample analysis to ensure that the LC-MS/MS system is functioning as intended. The results should meet the following minimum acceptance criteria for system performance, or the samples cannot be injected, unless a valid scientific reason is observed and documented (e.g., *signal : noise ratio ≥20:1, peak height*). 

When conducting stock or working solution stability assessments: 

A neat *XXX ng/mL* solution of *MK-XXXX* will be injected prior to initiating stock and working solution stability sample analysis to ensure that the LC-MS/MS system is functioning as intended. *Describe procedure for preparation.* 




text: 


"""
gpt_prompt = gpt_prompt + system_suitability


response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=gpt_prompt,
  temperature=0.0,
  max_tokens=1000,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0,
)


print(response['choices'][0]['text'])

filename = "BP0001_para1_chatgpt_output.csv"
rows = response['choices'][0]['text'].strip().split("\n")

# Split each row into a list of columns using the colon separator
#data = [row.split(": ") for row in rows]

data_list = []
with open(filename, "a", newline='') as csvfile:
  csvwriter = csv.writer(csvfile)
  for row in rows:
    # Split row into key and value
    row_split = row.split(": ", 1)
    if len(row_split) < 2:
        key = row_split[0]
        data_list.append({key: ""})
    else:
        # If value is < 2, there was no colon in that line and
        key, value = row_split
        data_list.append({key.strip(): value})
    csvwriter.writerow([key, value])



csv_file_name = 'BP0001_para1_chatgpt_output.csv'

with open(csv_file_name, 'r') as f:
    csv_content = f.read()

variables_without_values = []

for line in csv_content.split('\n'):
    if re.search(r'MK-\d+', line):
        mk_number = re.findall(r'MK-\d+', line)[0]
        print(f'MK Number: {mk_number}')
    elif re.search(r'Storage temperature', line, re.IGNORECASE):
        temperature_dilutent = re.findall(r'(\+|-)\d+oC', line)[0]
        print(f'Temperature Dilutent: {temperature_dilutent}')
    elif re.search(r'Diluent', line, re.IGNORECASE):
        dilutent = line.split(',')[1]
        print(f'Dilutent: {dilutent}')
    # Additional regex patterns for new variables
    elif re.search(r'Extraction method', line, re.IGNORECASE):
        extraction_method = line.split(',')[1]
        print(f'Extraction Method: {extraction_method}')
    elif re.search(r'Turbo ionspray', line, re.IGNORECASE):
        turbo_ionspray = line.split(',')[1]
        print(f'Turbo Ionspray: {turbo_ionspray}')
    elif re.search(r'Ion mode', line, re.IGNORECASE):
        polarity = line.split(',')[1]
        print(f'Polarity: {polarity}')
    elif re.search(r'MRM transition from drug', line, re.IGNORECASE):
        drug_lower_mz, drug_upper_mz = re.findall(r'\d+\.\d+', line)
        print(f'Drug Lower MZ: {drug_lower_mz}, Drug Upper MZ: {drug_upper_mz}')
    elif re.search(r'MRM transition for internal standard', line, re.IGNORECASE):
        is_lower_mz, is_upper_mz = re.findall(r'\d+\.\d+', line)
        print(f'IS Lower MZ: {is_lower_mz}, IS Upper MZ: {is_upper_mz}')
    elif re.search(r'LLOQ', line, re.IGNORECASE):
        lloq = float(line.split(',')[1].split()[0])
        print(f'LLOQ: {lloq}')
    elif re.search(r'1/x\^2 regression model', line, re.IGNORECASE):
        regression_model = line.split(',')[1]
        print(f'Regression Model: {regression_model}')
    elif re.search(r'Anticoagulant', line, re.IGNORECASE):
        anticoagulant = line.split(',')[1]
        print(f'Anticoagulant: {anticoagulant}')
    elif re.search(r'Study sample temperature', line, re.IGNORECASE):
        anticoagulant_temperature = line.split(',')[1]
        print(f'Anticoagulant Temperature: {anticoagulant_temperature}')



variables = {
    'bp_number': bp_number, 'extraction_method': extraction_method, 'mk_number': mk_number,
    'turbo_ionspray': turbo_ionspray, 'polarity': polarity, 'drug_lower_mz': drug_lower_mz,
    'drug_upper_mz': drug_upper_mz, 'is_lower_mz': is_lower_mz, 'is_upper_mz': is_upper_mz,
    'lloq': lloq, 'regression_model': regression_model, 'calibration_range_lower': calibration_range_lower,
    'calibration_range_upper': calibration_range_upper, 'calibration_sample_volume': calibration_sample_volume,
    'dilutent': dilutent, 'temperature_dilutent': temperature_dilutent, 'anticoagulant': anticoagulant,
    'anticoagulant_temperature': anticoagulant_temperature, 'special_requirments': special_requirments,
    'author': author
}

for key, value in variables.items():
    if value is None:
        variables_without_values.append(key)

print("Variables without values:", variables_without_values)



# Global Variables [text and tables]
global text
text = ''
global table_list
table_list = []

# Set up the Textract client
textract = boto3.client('textract', region_name='us-east-2')

# Specify the name of your PDF file and the S3 bucket to upload it to
pdf_file_name = 'BP-0002.pdf'
s3_bucket_name = 'merckbucket-123'

# Upload the PDF file to S3
with open(pdf_file_name, 'rb') as f:
    s3 = boto3.resource('s3')
    s3.Bucket(s3_bucket_name).put_object(Key=pdf_file_name, Body=f)

# Start the Textract job
response = textract.start_document_text_detection(
    DocumentLocation={
        'S3Object': {
            'Bucket': s3_bucket_name,
            'Name': pdf_file_name
        }
    }
)

# Get the job ID
job_id = response['JobId']

# Wait for the job to complete
while True:
    # Check if the job is complete
    response = textract.get_document_text_detection(JobId=job_id)
    if response['JobStatus'] == 'SUCCEEDED':
        break
    elif response['JobStatus'] == 'FAILED':
        raise Exception('Textract job failed')

    # Wait before checking the job status again
    time.sleep(5)

# Collect the pages of the document
pages = []

response = textract.get_document_text_detection(JobId=job_id)
pages.append(response)

while 'NextToken' in response:
    response = textract.get_document_text_detection(JobId=job_id, NextToken=response['NextToken'])
    pages.append(response)

# Print the detected text
for page in pages:
    for block in page['Blocks']:
        if block['BlockType'] == 'LINE':
            # print(block['Text'])
            text += block['Text'] + '\n'

# Read the PDF file
pdf_path = "BP-0002.pdf"

# Convert each page of the PDF file to PNG images
images = convert_from_path(pdf_path)

# Loop through each image and call Textract to analyze for tables
for i, image in enumerate(images):
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    response = textract.analyze_document(Document={'Bytes': image_bytes.getvalue()},
                                         FeatureTypes=["TABLES"])


    def map_blocks(blocks, block_type):
        return {
            block['Id']: block
            for block in blocks
            if block['BlockType'] == block_type
        }


    blocks = response['Blocks']
    tables = map_blocks(blocks, 'TABLE')
    cells = map_blocks(blocks, 'CELL')
    words = map_blocks(blocks, 'WORD')
    selections = map_blocks(blocks, 'SELECTION_ELEMENT')


    def get_children_ids(block):
        for rels in block.get('Relationships', []):
            if rels['Type'] == 'CHILD':
                yield from rels['Ids']


    dataframes = []

    for table in tables.values():

        # Determine all the cells that belong to this table
        table_cell_ids = []
        for cell in table.get('Relationships', []):
            if cell['Type'] == 'CHILD':
                table_cell_ids.extend(cell['Ids'])
        table_cells = [cells[cell_id] for cell_id in table_cell_ids]

        # Determine the table's number of rows and columns
        n_rows = max(cell['RowIndex'] for cell in table_cells)
        n_cols = max(cell['ColumnIndex'] for cell in table_cells)
        content = [[None for _ in range(n_cols)] for _ in range(n_rows)]

        # Fill in each cell
        for cell in table_cells:
            cell_contents = [
                words[child_id]['Text']
                if child_id in words
                else selections[child_id]['SelectionStatus']
                for child_id in get_children_ids(cell)
            ]
            i = cell['RowIndex'] - 1
            j = cell['ColumnIndex'] - 1
            content[i][j] = ' '.join(cell_contents)

        # We assume that the first row corresponds to the column names
        dataframe = pd.DataFrame(content[1:], columns=content[0])
        dataframes.append(dataframe)

        # Print the extracted tables
        # print('Tables:')
        for df in dataframes:
            # print(df)
            table_list.append(df)

        # print('Tables:' + str(len(dataframes)))

# Text Global Variable (Testing)
#print(text)

count = 1

# Table Global Variable (Testing)
for df in table_list:
    count += 1
    print("Table " + str(count))
    #print(df)os.environ["AWS_SECRET_ACCESS_KEY"] = ""


# This is because we have repeat tables

unique_tables = {}
for table in table_list:
    key = table.columns[0]
    if 'Analyte' in key:
        key = table.iloc[0, 0]
    if key not in unique_tables:
        unique_tables[key] = table

unique_df_list = list(unique_tables.values())

count1 = 0

count2 = 0

count3 = 0

for table in table_list:
  if (table.columns[0] == 'Category'):
    if (count2 == 1):
      unique_df_list.append(table)
    count2+=1
  if (table.columns[1] == 'Double Dilutions'):
    if (count3 == 1):
      unique_df_list.append(table)
    count3+=1

for df in unique_df_list:
  count1+=1
  #print("Table " + str(count1))
  #print(df)





# Extracts Values Given List of Rows and Columns [Named After Column]
def findColumnRowValues(searchTable, searchColumns, searchRows, globalVars):
    rowNumberList = []
    columnNumberList = []

    # Gets Position of Column
    for column in range(len(searchTable.columns)):
        for columnValue in searchColumns:
            if columnValue in searchTable.columns[column]:
                columnNumberList.append(column)

    # Gets Position of Row
    for row in range(len(searchTable)):
        for rowValue in searchRows:
            if rowValue in searchTable.iloc[row, 0]:
                rowNumberList.append(row)

    some = 0

    # Iterates over Positions and Sets Global Variables
    for i, row in enumerate(rowNumberList):
        for j, column in enumerate(columnNumberList):
            globalVars[some] = searchTable.iloc[row, column]
            some += 1

    return globalVars


# Get and label values from Rows under Header [Analyte] - Labeling not Based on Column 1
def findAllValuesUnderColumn(searchTable, globalVars):
    for row in range(len(searchTable)):
        for column in range(len(searchTable.columns)):
            if column < len(globalVars):
                globalVars[column] = searchTable.iloc[row, column]
        return globalVars


count4 = 0
count5 = 0

for table in unique_df_list:
    if "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and "Mass Spectrometer" in re.sub(' +',' ', table.iloc[0,0].replace("\n", "")):
        #print("a")
        columnValues = ["Components"]
        rowValues = ["Mass Spectrometer", "Liquid Handling"]
        globalVarValues = [g_massspec_component, g_liquidhandling_component]
        returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
        g_massspec_component = returnValue[0]
        g_liquidhandling_component = returnValue[1]
    elif "Category (General)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("b")
        l = len(table.to_numpy())
        column_category = table.to_numpy()[l-1][0]
        column_manufacturer = table.to_numpy()[l-1][1]
    elif "Category (Equipment)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("c")
        columnValues = ["Manufacturer"]
        rowValues = ["Microbalance", "Analytical Balance", "Refrigerated Centrifuge", "pH Meter", "Plate sealer"]
        globalVarValues = [microbalance_manufacturer, analytical_balance_manufacturer, refrigerated_centrifuge_manufacturer, pH_meter_manufacturer, platesealer_manufacturer]
        returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
        #print(returnValue)
        microbalance_manufacturer = returnValue[0]
        analytical_balance_manufacturer = returnValue[1]
        pH_meter_manufacturer = returnValue[2]
    elif "Category (Pipettes)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("d")
        columnValues = ["Manufacturer"]
        rowValues = ["Adjustable Pipettes", "Pipette Tips"]
        globalVarValues = [adjustable_pipettes_manufacturer, pipette_tips_manufacturer]
        returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
        #print(returnValue)
        adjustable_pipettes_manufacturer = returnValue[0]
        pipette_tips_manufacturer = returnValue[1]
    elif "Category (Automation Supplies)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        # print("e")
        columnValues = ["Manufacturer"]
        rowValues = ["Reagent", "Automated"]
        globalVarValues = [reagent_troughs_provider, automated_workstation_tips_provider]
        returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
        #print(returnValue)
        reagent_troughs_provider = returnValue[0]
        automated_workstation_tips_provider = returnValue[1]
    elif "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and "Analyte / L-Number" in re.sub(' +',' ', table.iloc[0,0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("f")
        columnValues = ["Parent", "Internal"]
        rowValues = ["Analyte", "Form", "Molecular Weight", "Watson ID"]
        globalVarValues = [analyte_l_parent, analyte_l_is, form_parent, form_is, molecular_weight_parent, molecular_weight_is, watson_id_parent, watson_id_is]
        returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
        analyte_l_parent = returnValue[0]
        analyte_l_is = returnValue[1]
        form_parent = returnValue[2]
        form_is = returnValue[3]
        molecular_weight_parent = returnValue[4]
        molecular_weight_is = returnValue[5]
        watson_id_parent = returnValue[6]
        watson_id_is = returnValue[7]
    elif "Species" in re.sub(' +',' ', table.columns[1].replace("\n", "")) and "Anticoagulant" in re.sub(' +',' ', table.columns[2].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("g")
        globalVarValues = [matrix, species, anticoagulant, supplier]
        returnValue = findAllValuesUnderColumn(table, globalVarValues)
        matrix = returnValue[0]
        species = returnValue[1]
        anticoagulant = returnValue[2]
        supplier = returnValue[3]
    elif "Standard Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("h")
        table_standard_solution_id = table
    elif "QC Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("i")
        table_qc_solution_id = table
    elif "QC ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("j")
        table_qc_id = table
    elif "Step" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("k")
        table_step = table
    elif "LC" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count5 == 0:
        count5+=1
        #print(str(table.iloc[0,0]))
        #print("l")
        #print(table)
        columnValue = ["Settings"]
        rowValue = ["Loop Option", "Elution", "Mobile Phase A", "Mobile Phase B"]
        globalVarValues = [loop_option_settings, elution_settings, mobile_phase_a_settings, mobile_phase_b_settings]
        returnValue = findColumnRowValues(table, columnValue, rowValue, globalVarValues)
        #print(returnValue)
        loop_option_settings = returnValue[0]
        elution_settings = returnValue[1]
        mobile_phase_a_settings = returnValue[2]
        mobile_phase_b_settings = returnValue[3]
    elif "MS" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("m")
        columnValue = ["Settings"]
        rowValue = ["Ion Source", "Ion Mode", "Ionization potential", "Temperature", "MR pause", "MS settling"]
        globalVarValues = [ion_source_settings, ion_mode_settings, ionization_potential_settings, temperature_settings, mr_pause_settings, mr_settling_time_settings]
        returnValue = findColumnRowValues(table, columnValue, rowValue, globalVarValues)
        #print(returnValue)
        ion_source_settings = returnValue[0]
        ion_mode_settings = returnValue[1]
        ionization_potential_settings = returnValue[2]
        temperature_settings = returnValue[3]
        mr_pause_settings = returnValue[4]
        mr_settling_time_settings = returnValue[5]
    elif "Ions Monitored" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
        #print(str(table.iloc[0,0]))
        #print("n")
        table_ions_monitored = table
    elif "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count4 == 0:
      #print(str(table.iloc[0,0]))
      #print("o")
      globalVarValues = [analyte_1, peak_height_1, retention_time_1]
      returnValue = findAllValuesUnderColumn(table, globalVarValues)
      analyte_1 = returnValue[0]
      peak_height_1 = returnValue[1]
      retention_time_1 = returnValue[2]
      count4+=1
    elif "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count4 == 1:
      #print(str(table.iloc[0,0]))
      #print("p")
      globalVarValues = [analyte_2, peak_height_2, retention_time_2]
      returnValue = findAllValuesUnderColumn(table, globalVarValues)
      analyte_2 = returnValue[0]
      peak_height_2 = returnValue[1]
      retention_time_2 = returnValue[2]


# Instrumentation
print("Mass Spectrometer: " + str(g_massspec_component))
print("Liquid Handling: "  + str(g_liquidhandling_component))

# Category (General)
print("Column: " + str(column_category))
print("Column Manufacturer: " + str(column_manufacturer))

# Category (Equipment)
print("Microbalance Manufacturer: " + str(microbalance_manufacturer))
print("Analytical Balance Manufacturer: " + str(analytical_balance_manufacturer))
print("Refrigerated Centrifuge Manufacturer: " + str(refrigerated_centrifuge_manufacturer))
print("pH Meter Manufacturer: " + str(pH_meter_manufacturer))
print("Plate Sealer Manufacturer: " + str(platesealer_manufacturer))

# Category (Pipettes)
print("Adjustable Pipettes Manufacturer: " + str(adjustable_pipettes_manufacturer))
print("Pipette tips Manufacturer: " + str(pipette_tips_manufacturer))

# Category (Automation Supplies)
print("Reagent Troughs Provider: " + str(reagent_troughs_provider))
print("Automated Workstation Tips: " + str(automated_workstation_tips_provider))

# Reference Standards
print("Analyte/L Parent: " + str(analyte_l_parent))
print("Analyte/L IS: " + str(analyte_l_is))
print("Form Parent: " + str(form_parent))
print("Form IS: " + str(form_is))
print("Molecular Weight Parent: " + str(molecular_weight_parent))
print("Molecular Weight IS: " + str(molecular_weight_is))
print("Watson ID Parent: " + str(watson_id_parent))
print("Watson ID IS: " + str(watson_id_is))

# Matrix/Species/Anticoagulant/Supplier
print("Matrix: " + str(matrix))
print("Species: " + str(species))
print("Anticoagulant: " + str(anticoagulant))
print("Supplier: " + str(supplier))

# UPLC/LC
print("Loop Option Settings: " + str(loop_option_settings))
print("Elution Settings: " + str(elution_settings))
print("Mobile Phase A Settings: " + str(mobile_phase_a_settings))
print("Mobile Phase B Settings: " + str(mobile_phase_b_settings))

# MS
print("Ion Source Settings: " + str(ion_source_settings))
print("Ion Mode Settings: " + str(ion_mode_settings))
print("Ionization Potential Settings: " + str(ionization_potential_settings))
print("Temperature Settings: " + str(temperature_settings))
print("MR Pause Settings: " + str(mr_pause_settings))
print("MR Settling Time Settings: " + str(mr_settling_time_settings))

# Analyte 1
print("Analyte 1: " + str(analyte_1))
print("Peak Height Analyte 1: " + str(peak_height_1))
print("Retention Time Analyte 1: " + str(retention_time_1))

# Analyte 2
print("Analyte 2: " + str(analyte_2))
print("Peak Height Analyte 2: " + str(peak_height_2))
print("Retention Time Analyte 2: " + str(retention_time_2))

# Other Important Tables
print("Table Standard Solution ID")
print(table_standard_solution_id)
print("Table QC Solution ID")
print(table_qc_solution_id)
print("Table QC ID")
print(table_qc_id)
print("Table Step")
print(table_step)
print("Table Ions Monitored")
print(table_ions_monitored)

# Non-table Global Variables


# ** Not going to make global variables for below sections since Merck keeps changing the document **

# MK-XXXX Standard Preparation
# MK-XXXX Quality Control
# IS Preparation
# Procedure


# CSV Insertion
import csv

# List of global variable names
global_var_names = ['g_massspec_component', 'g_liquidhandling_component', 'column_category', 'column_manufacturer', 'microbalance_manufacturer', 'analytical_balance_manufacturer', 'refrigerated_centrifuge_manufacturer', 'pH_meter_manufacturer', 'platesealer_manufacturer', 'adjustable_pipettes_manufacturer', 'pipette_tips_manufacturer', 'reagent_troughs_provider', 'automated_workstation_tips_provider', 'analyte_l_parent', 'analyte_l_is', 'form_parent', 'form_is', 'molecular_weight_parent', 'molecular_weight_is', 'watson_id_parent', 'watson_id_is', 'matrix', 'species', 'anticoagulant', 'supplier', 'loop_option_settings', 'elution_settings', 'mobile_phase_a_settings', 'mobile_phase_b_settings', 'ion_source_settings', 'ion_mode_settings', 'ionization_potential_settings', 'temperature_settings', 'mr_pause_settings', 'mr_settling_time_settings', 'analyte_1', 'peak_height_1', 'retention_time_1', 'analyte_2', 'peak_height_2', 'retention_time_2', 'table_standard_solution_id', 'table_qc_solution_id', 'table_qc_id', 'table_step', 'table_ions_monitored', 'bp_number', 'extraction_method', 'mk_number', 'turbo_ionspray', 'polarity', 'drug_lower_m/z', 'drug_upper_m/z', 'is_lower_m/z', 'is_upper_m/z', 'lloq', 'regression_model', 'calibration_range_lower', 'calibration_range_upper', 'calibration_sample_volume', 'dilutent', 'temperature_dilutent', 'anticoagulant_temperature', 'special_requirments', 'author']

# List of global variables
global_vars = [g_massspec_component, g_liquidhandling_component, column_category, column_manufacturer, microbalance_manufacturer, analytical_balance_manufacturer, refrigerated_centrifuge_manufacturer, pH_meter_manufacturer, platesealer_manufacturer, adjustable_pipettes_manufacturer, pipette_tips_manufacturer, reagent_troughs_provider, automated_workstation_tips_provider, analyte_l_parent, analyte_l_is, form_parent, form_is, molecular_weight_parent, molecular_weight_is, watson_id_parent, watson_id_is, matrix, species, anticoagulant, supplier, loop_option_settings, elution_settings, mobile_phase_a_settings, mobile_phase_b_settings, ion_source_settings, ion_mode_settings, ionization_potential_settings, temperature_settings, mr_pause_settings, mr_settling_time_settings, analyte_1, peak_height_1, retention_time_1, analyte_2, peak_height_2, retention_time_2, table_standard_solution_id, table_qc_solution_id, table_qc_id, table_step, table_ions_monitored, bp_number, extraction_method, mk_number, turbo_ionspray, polarity, drug_lower_mz, drug_upper_mz, is_lower_mz, is_upper_mz, lloq, regression_model, calibration_range_lower, calibration_range_upper, calibration_sample_volume, dilutent, temperature_dilutent, anticoagulant_temperature, special_requirments, author]

# Writing to CSV file
with open('global_vars.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(global_var_names)
    writer.writerow(global_vars)

searchColumns = ["Settings"]
searchRows = ["Loop Option", "Elution", "Mobile Phase A", "Mobile Phase B"]
globalVarValues = [loop_option_settings, elution_settings, mobile_phase_a_settings, mobile_phase_b_settings]
searchTable = unique_df_list[4]

rowNumberList = []
columnNumberList = []

# Gets Position of Column
for column in range(len(searchTable.columns)):
    for columnValue in searchColumns:
        print(searchTable.columns[column])
        if columnValue in searchTable.columns[column]:
            print("found")
            columnNumberList.append(column)

# Gets Position of Row
for row in range(len(searchTable)):
    for rowValue in searchRows:
        print(searchTable.iloc[row, 0])
        if rowValue in searchTable.iloc[row, 0]:
            print("found")
            rowNumberList.append(row)

count = 0

print(rowNumberList)
print(columnNumberList)

# Iterates over Positions and Sets Global Variables
for i, row in enumerate(rowNumberList):
    for j, column in enumerate(columnNumberList):
        print(searchTable.iloc[row, column])
        globalVarValues[count] = searchTable.iloc[row, column]
        count += 1

print(globalVarValues)


# Gets Position of Column
for column in range(len(searchTable.columns)):
  for columnValue in searchColumns:
    if columnValue in searchTable.columns[column]:
      columnNumberList.append(column)

# Gets Position of Row
for row in range(len(searchTable)):
  for rowValue in searchRows:
    if rowValue in searchTable.iloc[row,0]:
      rowNumberList.append(row)

count = 0

# Iterates over Positions and Sets Global Variables
for i, row in enumerate(rowNumberList):
  for j, column in enumerate(columnNumberList):
      globalVars[count] = searchTable.iloc[row, column]
      count+=1

nodes = []
relationships = []
#Section 0
bp_node = Node('BP', {'BP_number': bp_number})
special_requirement_node = Node('SpecialRequirement', {'Requirements': special_requirments})
instrumentationListNode = Node("Instrumentation",
{"Instrumentation": bp_number, "Mass Spectrometer": g_massspec_component, "LC": bp_number,
"Liquid Handling": g_liquidhandling_component, "Manufacturer": column_manufacturer, "LCcomponent": None})

reagentsListNode = Node("Reagents", {"Properties": {}})
solutionsListNode = Node("Solutions", {"Properties": {}})
calculationParametersListNode = Node("Calculation Parameters", {"Model": regression_model})
#section 1
instrumentationNode = Node("Instrumentation", {'BP_number': bp_number})
massSpectrometerNode = Node("MassSpectrometer", {"Name": g_massspec_component})
LCNode = Node("LC", {"BP Number": bp_number})
liquidHandlingNode = Node("LiquidHandling", {"Name": g_liquidhandling_component})
manufacturerNode = Node("Manufacturer", {"Name": column_manufacturer})
LCComponentNode = Node("LC Component", {"Name": None})
#section 2
suppliesListNode = Node("Supplies", {"BP_Number": bp_number})
equipmentNode = Node("Equipment", {"Microbalance Manufacturer": microbalance_manufacturer,
                               "Analytical Balance Manufacturer": analytical_balance_manufacturer,
                               "Refrigerated Centrifuge Manufacturer": refrigerated_centrifuge_manufacturer,
                               "pH Meter Manufacturer": pH_meter_manufacturer,
                               "Platesealer Manufacturer": platesealer_manufacturer})
pipetteNode = Node("Pipettes", {"Adjustable Pipettes Manufacturer": adjustable_pipettes_manufacturer,
"Pipette Tips Manufacturer": pipette_tips_manufacturer})
automationSuppNode = Node("Automation Supplies", {"Reagent Troughs Provider": reagent_troughs_provider,
"Automated Workstation Tips Provider": automated_workstation_tips_provider})
#Section 3
compoundNode = Node("Compound", {"Analyte/L-Number": analyte_l_parent, "Form": form_parent, "Molecular_Weight": molecular_weight_parent, "Watson_ID": watson_id_parent})
epimerNode = Node("L-number", {"Property": analyte_l_parent})
#Section 4
biologicalMatrixListNode = Node("BiologicalMatrix", {
    "BP_number": bp_number,
    "Matrix": matrix,
    "Species": species,
    "Anticoagulant": anticoagulant,
    "Extraction Method": extraction_method,
    "Storage_temp": anticoagulant_temperature
})
manufacturerNode = Node("Manufacturer", {"Name": None})
#Sections 5,6 no red text
#Section 7
StandardPreparationNode = Node("BP Number", {"Property": bp_number})
mixedIntermediateStandardSolutionNode = Node("Mixed Intermediate Standard Solution", {"Table": table_standard_solution_id})
stockStandardSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
workingStandardSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
#Section 8
qcPreparationListNode = Node("QC Preparation", {"BP Number": bp_number})
stockQCSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
workingQCSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
workingQCTableNode = Node("Working QC Solution ID", {"Table": table_qc_solution_id})
matrixQCTableNode = Node("Matrix QC ID", {"Table": table_qc_id})
matrixQCNode = Node("Matrix QC", {"Table": lloq})
#Section 9
ISPreparationListNode = Node("ISPreparation", {"BP_number": bp_number})
stockISSolutionNode = Node("StockISSolution", {
    "Name": mk_number,
    "PreparationSummary": special_requirments,
    "Use": None,
    "Storage": temperature_dilutent
})
workingISSolutionNode = Node("WorkingISSolution", {"Name": mk_number})
#Section 10 (None)
#Section 12
operatingParametersNode = Node("Operating Parameters", {"BP Number": bp_number})
UPLCParametersNode = Node("UPLCParameters", {
    "Column": column_category,
    "Loop_Option": loop_option_settings,
    "Elution": elution_settings,
    "Mobile_Phase_A": mobile_phase_a_settings,
    "Mobile_Phase_B": mobile_phase_b_settings,
})

UPLCProfileNode = Node("UPLCProfile", {
    "Levels": None,
    "Time (min)": None,
    "%A": None,
    "%B": None,
    "Curve": None
})

MSParametersNode = Node("MSParameters", {
    "Ion_Source": ion_source_settings,
    "Ion_Mode": ion_mode_settings,
    "Q1/Q3_Resolutions": None,
    "Ionization_potential (IS)": ionization_potential_settings,
    "Temperature": temperature_settings,
    "MR pause between mass range": mr_pause_settings,
    "MS settling time": mr_settling_time_settings
})
#Section 14

nodes.append(bp_node)
nodes.append(special_requirement_node)
nodes.append(compoundNode)
nodes.append(instrumentationNode)
nodes.append(massSpectrometerNode)
nodes.append(LCNode)
nodes.append(liquidHandlingNode)
nodes.append(manufacturerNode)
nodes.append(LCComponentNode)
nodes.append(instrumentationListNode)
nodes.append(biologicalMatrixListNode)
nodes.append(reagentsListNode)
nodes.append(solutionsListNode)
nodes.append(qcPreparationListNode)
nodes.append(calculationParametersListNode)
nodes.append(StandardPreparationNode)
nodes.append(mixedIntermediateStandardSolutionNode)
nodes.append(stockStandardSolutionNode)
nodes.append(workingStandardSolutionNode)
nodes.append(stockQCSolutionNode)
nodes.append(workingQCSolutionNode)
nodes.append(workingQCTableNode)
nodes.append(matrixQCTableNode)
nodes.append(operatingParametersNode)
nodes.append(UPLCParametersNode)
nodes.append(MSParametersNode)
nodes.append(suppliesListNode)
nodes.append(equipmentNode)
nodes.append(pipetteNode)
nodes.append(automationSuppNode)
nodes.append(ISPreparationListNode)
nodes.append(stockISSolutionNode)
nodes.append(workingISSolutionNode)
nodes.append(UPLCParametersNode)
nodes.append(UPLCProfileNode)
nodes.append(MSParametersNode)
relationships.append(Relationship(operatingParametersNode, UPLCParametersNode, 'HAS_A', {}))
relationships.append(Relationship(operatingParametersNode, UPLCProfileNode, 'HAS_A', {}))
relationships.append(Relationship(operatingParametersNode, MSParametersNode, 'HAS_A', {}))
nodes.append(epimerNode)
nodes.append(matrixQCNode)




relationships.append(Relationship(bp_node, special_requirement_node, 'HAS', {}))
relationships.append(Relationship(bp_node, compoundNode, 'ANALYZES', {}))
relationships.append(Relationship(instrumentationNode, massSpectrometerNode, 'USES', {}))
relationships.append(Relationship(manufacturerNode, massSpectrometerNode, 'SUPPLIES', {}))
relationships.append(Relationship(instrumentationNode, LCNode, 'USES', {}))
relationships.append(Relationship(manufacturerNode, LCNode, 'SUPPLIES', {}))
relationships.append(Relationship(LCNode, LCComponentNode, "HAS_A", {}))
relationships.append(Relationship(instrumentationNode, liquidHandlingNode, "USES", {}))
relationships.append(Relationship(manufacturerNode, liquidHandlingNode, "SUPPLIES", {}))
relationships.append(Relationship(bp_node, instrumentationNode, "USES", {}))
relationships.append(Relationship(bp_node, suppliesListNode, 'USES', {}))
relationships.append(Relationship(bp_node, biologicalMatrixListNode, 'USES', {}))
relationships.append(Relationship(bp_node, reagentsListNode, 'USES', {}))
relationships.append(Relationship(bp_node, solutionsListNode, 'USES', {}))
relationships.append(Relationship(bp_node, StandardPreparationNode, 'USES', {}))
relationships.append(Relationship(bp_node, qcPreparationListNode, 'USES', {}))
relationships.append(Relationship(bp_node, ISPreparationListNode, 'USES', {}))
relationships.append(Relationship(bp_node, calculationParametersListNode, 'USES', {}))
relationships.append(Relationship(StandardPreparationNode, stockStandardSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(StandardPreparationNode, workingStandardSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(StandardPreparationNode, mixedIntermediateStandardSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(qcPreparationListNode, stockQCSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(qcPreparationListNode, workingQCSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(qcPreparationListNode, matrixQCNode, 'HAS_A', {}))
relationships.append(Relationship(qcPreparationListNode, workingQCTableNode, 'HAS_A', {}))
relationships.append(Relationship(qcPreparationListNode, matrixQCTableNode, 'HAS_A', {}))
relationships.append(Relationship(operatingParametersNode, UPLCParametersNode, 'HAS_A', {}))
relationships.append(Relationship(operatingParametersNode, MSParametersNode, 'HAS_A', {}))
relationships.append(Relationship(operatingParametersNode, UPLCProfileNode, 'HAS_A', {}))


relationships.append(Relationship(ISPreparationListNode, stockISSolutionNode, 'HAS_A', {}))
relationships.append(Relationship(ISPreparationListNode, workingISSolutionNode, 'HAS_A', {}))



