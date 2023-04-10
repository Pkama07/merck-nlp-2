import boto3
import csv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from GraphManager import GraphManager
import io
from node import Node
import openai
import os
import pandas as pd
from pdf2image import convert_from_path
import re
from relationship import Relationship
import time


app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
s3_bucket_name = 'merckbucket-123'

s3 = boto3.resource("s3")
textract = boto3.client('textract', region_name='us-east-2')

gm = GraphManager("neo4j+ssc://39f470cd.databases.neo4j.io", "neo4j","GRQHnkLdqja2PXzjYUg4wwoCxCI3uAGPOjbe6N_K6KM")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def upload_s3(filepath):
    s3 = boto3.resource("s3")
    with open(filepath, "rb") as file:
        s3.Bucket(s3_bucket_name).put_object(Key=filepath[5:], Body=file)


def parse_pdf(filename):
    response = textract.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': filename
            }
        }
    )
    job_id = response['JobId']
    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        if response['JobStatus'] == 'SUCCEEDED':
            break
        elif response['JobStatus'] == 'FAILED':
            raise Exception('Textract job failed')
        print("waiting")
        time.sleep(3)
    pages = []
    response = textract.get_document_text_detection(JobId=job_id)
    pages.append(response)
    while 'NextToken' in response:
        response = textract.get_document_text_detection(JobId=job_id, NextToken=response['NextToken'])
        pages.append(response)
    text = ""
    for page in pages:
        for block in page['Blocks']:
            if block['BlockType'] == 'LINE':
                text += block['Text'] + '\n'
    return text


def map_blocks(blocks, block_type):
    return {
        block['Id']: block
        for block in blocks
        if block['BlockType'] == block_type
    }


def get_children_ids(block):
    for rels in block.get('Relationships', []):
        if rels['Type'] == 'CHILD':
            yield from rels['Ids']


def generate_tables(filepath):
    images = convert_from_path(filepath)
    table_list = []
    for i, image in enumerate(images):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        response = textract.analyze_document(Document={'Bytes': image_bytes.getvalue()},
                                            FeatureTypes=["TABLES"])
        blocks = response['Blocks']
        tables = map_blocks(blocks, 'TABLE')
        cells = map_blocks(blocks, 'CELL')
        words = map_blocks(blocks, 'WORD')
        selections = map_blocks(blocks, 'SELECTION_ELEMENT')
        dataframes = []

        for table in tables.values():
            table_cell_ids = []
            for cell in table.get('Relationships', []):
                if cell['Type'] == 'CHILD':
                    table_cell_ids.extend(cell['Ids'])
            table_cells = [cells[cell_id] for cell_id in table_cell_ids]
            n_rows = max(cell['RowIndex'] for cell in table_cells)
            n_cols = max(cell['ColumnIndex'] for cell in table_cells)
            content = [[None for _ in range(n_cols)] for _ in range(n_rows)]
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
            dataframe = pd.DataFrame(content[1:], columns=content[0])
            dataframes.append(dataframe)
            for df in dataframes:
                table_list.append(df)
    unique_tables = {}
    for table in table_list:
        key = table.columns[0]
        if 'Analyte' in key:
            key = table.iloc[0, 0]
        if key not in unique_tables:
            unique_tables[key] = table
    unique_df_list = list(unique_tables.values())
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
    return unique_df_list


def findColumnRowValues(searchTable, searchColumns, searchRows, globalVars):
    rowNumberList = []
    columnNumberList = []
    for column in range(len(searchTable.columns)):
        for columnValue in searchColumns:
            if columnValue in searchTable.columns[column]:
                columnNumberList.append(column)
    for row in range(len(searchTable)):
        for rowValue in searchRows:
            if rowValue in searchTable.iloc[row, 0]:
                rowNumberList.append(row)
    some = 0
    for i, row in enumerate(rowNumberList):
        for j, column in enumerate(columnNumberList):
            globalVars[some] = searchTable.iloc[row, column]
            some += 1
    return globalVars


def findAllValuesUnderColumn(searchTable, globalVars):
    for row in range(len(searchTable)):
        for column in range(len(searchTable.columns)):
            if column < len(globalVars):
                globalVars[column] = searchTable.iloc[row, column]
        return globalVars


def generate_DB_content(searchString, tables, bp_number):
    extraction_method = None
    mk_number = None
    turbo_ionspray = None
    polarity = None
    drug_lower_mz = None
    drug_upper_mz = None
    is_lower_mz = None
    is_upper_mz = None
    lloq = None
    regression_model = None
    calibration_range_lower = None
    calibration_range_upper = None
    calibration_sample_volume = None
    dilutent = None
    temperature_dilutent = None
    anticoagulant = None
    anticoagulant_temperature = None
    special_requirements = None
    author = None
    g_massspec_component = None
    g_liquidhandling_component = None
    column_category = None
    column_manufacturer = None
    microbalance_manufacturer = None
    analytical_balance_manufacturer = None
    refrigerated_centrifuge_manufacturer = None
    pH_meter_manufacturer = None
    platesealer_manufacturer = None
    adjustable_pipettes_manufacturer = None
    pipette_tips_manufacturer = None
    reagent_troughs_provider = None
    automated_workstation_tips_provider = None
    analyte_l_parent = None
    analyte_l_is = None
    form_parent = None
    form_is = None
    molecular_weight_parent = None
    molecular_weight_is = None
    watson_id_parent = None
    watson_id_is = None
    matrix = None
    species = None
    anticoagulant = None
    supplier = None
    loop_option_settings = None
    elution_settings = None
    mobile_phase_a_settings = None
    mobile_phase_b_settings = None
    ion_source_settings = None
    ion_mode_settings = None
    ionization_potential_settings = None
    temperature_settings = None
    mr_pause_settings = None
    mr_settling_time_settings = None
    analyte_1 = None
    peak_height_1 = None
    retention_time_1 = None
    analyte_2 = None
    peak_height_2 = None
    retention_time_2 = None
    table_standard_solution_id = None
    table_qc_solution_id = None
    table_qc_id = None
    table_step = None
    table_ions_monitored = None
    page1 = None
    qc_sample_prep = None
    internal_standard = None
    procedure = None
    system_suitability = None
    match = re.search(r'^.*?Page 2', searchString, re.DOTALL)
    if match:
        page1 = match.group(0)
    match1 = re.search(r"SAMPLE PREPARATION(.*?)INTERNAL STANDARD", searchString, re.I)
    qc_sample_prep = match1.group(1)
    match2 = re.search(r"INTERNAL STANDARD(.*?)OPERATING PARAMETERS", searchString, re.I)
    internal_standard = match2.group(1)
    match3 = re.search(r"PROCEDURE(.*?)OPERATING PARAMETERS", searchString, re.I)
    procedure = match3.group(1)
    match4 = re.search(r"SYSTEM SUITABILITY(.*?)SOFTWARE AND CALCULATION", searchString, re.I)
    system_suitability = match4.group(1)
    prompts = []
    prompts.append(""" Extract the BP Number, extraction method, the species, drug, internal standard, chromatography type, interface, ion mode, MRM transition from drug, MRM transition for internal standard, LLOQ, 1/x^2 regression model, calibration range, anticoagulant, study samples, study sample temperature  given the context below?: \n
    context:""" + page1)
    prompts.append(""" Use the text surrounded by the ** to extract the values from the following text with a similar structure.
    The glassware, reagents, and dilutions described in this procedure should serve as a guide and may be substituted for their equivalent when necessary to obtain similar results. 
    Blank *matrix*: 
    Thaw and *centrifuge for 5 minutes at 2800 rpm* 
    *Stabilize blank matrix daily (if applicable)* 
    Describe storage conditions of treated matrix e.g., store on wet ice until ready to use 
    Biological *matrix* samples (e.g., unknown matrix samples, matrix QCs): 
    Allow samples to thaw completely 
    *Describe vortexing, rotating and centrifuge conditions <<Vortex samples for x minute at setting x>> <<Rotate sample rack on lab rotator for 10 minutes at setting 40>> <<Centrifuge for 5 minutes at 4000 rpm>> *
    text: \n
    """ + procedure)
    prompts.append(""" 
    Use the text surrounded by the ** to extract the values from the following text with a similar structure. When you extract can you output as a name:value format please. and for the describe proceducure store it as procedure: all the bulletpoints that correlate. MK-XXXX is MK Number. Neat solution is the ng/ml.
    When assaying biological extracts: 
    An extracted system suitability sample at the LLOQ will be injected prior to sample analysis to ensure that the LC-MS/MS system is functioning as intended. The results should meet the following minimum acceptance criteria for system performance, or the samples cannot be injected, unless a valid scientific reason is observed and documented (e.g., *signal : noise ratio ≥20:1, peak height*). 
    When conducting stock or working solution stability assessments: 
    A neat *XXX ng/mL* solution of *MK-XXXX* will be injected prior to initiating stock and working solution stability sample analysis to ensure that the LC-MS/MS system is functioning as intended. *Describe procedure for preparation.* 
    text: 
    """ + system_suitability)
    filename = "chatgpt_output.csv"
    print("starting the chatgpt prompts")
    for prompt in prompts:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            temperature=0.0,
            max_tokens=400,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        rows = response['choices'][0]['text'].strip().split("\n")
        with open(filename, "a", newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            for row in rows:
                row_split = row.split(": ", 1)
                if len(row_split) < 2:
                    key = row_split[0]
                else:
                    key, value = row_split
                csvwriter.writerow([key, value])
    with open(filename, 'r') as f:
        csv_content = f.read()
    os.remove(filename)
    for line in csv_content.split('\n'):
        if re.search(r'MK-\d+', line):
            mk_number = re.findall(r'MK-\d+', line)[0]
        elif re.search(r'Storage temperature', line, re.IGNORECASE):
            temperature_dilutent = re.findall(r'(\+|-)\d+oC', line)[0]
        elif re.search(r'Diluent', line, re.IGNORECASE):
            dilutent = line.split(',')[1]
        elif re.search(r'Extraction method', line, re.IGNORECASE):
            extraction_method = line.split(',')[1]
        elif re.search(r'Turbo ionspray', line, re.IGNORECASE):
            turbo_ionspray = line.split(',')[1]
        elif re.search(r'Ion mode', line, re.IGNORECASE):
            polarity = line.split(',')[1]
        elif re.search(r'MRM transition from drug', line, re.IGNORECASE):
            drug_lower_mz, drug_upper_mz = re.findall(r'\d+\.\d+', line)
        elif re.search(r'MRM transition for internal standard', line, re.IGNORECASE):
            is_lower_mz, is_upper_mz = re.findall(r'\d+\.\d+', line)
        elif re.search(r'LLOQ', line, re.IGNORECASE):
            lloq = float(line.split(',')[1].split()[0])
        elif re.search(r'1/x\^2 regression model', line, re.IGNORECASE):
            regression_model = line.split(',')[1]
        elif re.search(r'Anticoagulant', line, re.IGNORECASE):
            anticoagulant = line.split(',')[1]
        elif re.search(r'Study sample temperature', line, re.IGNORECASE):
            anticoagulant_temperature = line.split(',')[1]
    count4 = 0
    count5 = 0
    for table in tables:
        if "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and "Mass Spectrometer" in re.sub(' +',' ', table.iloc[0,0].replace("\n", "")):
            columnValues = ["Components"]
            rowValues = ["Mass Spectrometer", "Liquid Handling"]
            globalVarValues = [g_massspec_component, g_liquidhandling_component]
            returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
            g_massspec_component = returnValue[0]
            g_liquidhandling_component = returnValue[1]
        elif "Category (General)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            l = len(table.to_numpy())
            column_category = table.to_numpy()[l-1][0]
            column_manufacturer = table.to_numpy()[l-1][1]
        elif "Category (Equipment)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            columnValues = ["Manufacturer"]
            rowValues = ["Microbalance", "Analytical Balance", "Refrigerated Centrifuge", "pH Meter", "Plate sealer"]
            globalVarValues = [microbalance_manufacturer, analytical_balance_manufacturer, refrigerated_centrifuge_manufacturer, pH_meter_manufacturer, platesealer_manufacturer]
            returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
            microbalance_manufacturer = returnValue[0]
            analytical_balance_manufacturer = returnValue[1]
            pH_meter_manufacturer = returnValue[2]
        elif "Category (Pipettes)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            columnValues = ["Manufacturer"]
            rowValues = ["Adjustable Pipettes", "Pipette Tips"]
            globalVarValues = [adjustable_pipettes_manufacturer, pipette_tips_manufacturer]
            returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
            adjustable_pipettes_manufacturer = returnValue[0]
            pipette_tips_manufacturer = returnValue[1]
        elif "Category (Automation Supplies)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            columnValues = ["Manufacturer"]
            rowValues = ["Reagent", "Automated"]
            globalVarValues = [reagent_troughs_provider, automated_workstation_tips_provider]
            returnValue = findColumnRowValues(table, columnValues, rowValues, globalVarValues)
            reagent_troughs_provider = returnValue[0]
            automated_workstation_tips_provider = returnValue[1]
        elif "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and "Analyte / L-Number" in re.sub(' +',' ', table.iloc[0,0].replace("\n", "")):
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
            globalVarValues = [matrix, species, anticoagulant, supplier]
            returnValue = findAllValuesUnderColumn(table, globalVarValues)
            matrix = returnValue[0]
            species = returnValue[1]
            anticoagulant = returnValue[2]
            supplier = returnValue[3]
        elif "Standard Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            table_standard_solution_id = table
        elif "QC Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            table_qc_solution_id = table
        elif "QC ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            table_qc_id = table
        elif "Step" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            table_step = table
        elif "LC" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count5 == 0:
            count5 += 1
            columnValue = ["Settings"]
            rowValue = ["Loop Option", "Elution", "Mobile Phase A", "Mobile Phase B"]
            globalVarValues = [loop_option_settings, elution_settings, mobile_phase_a_settings, mobile_phase_b_settings]
            returnValue = findColumnRowValues(table, columnValue, rowValue, globalVarValues)
            loop_option_settings = returnValue[0]
            elution_settings = returnValue[1]
            mobile_phase_a_settings = returnValue[2]
            mobile_phase_b_settings = returnValue[3]
        elif "MS" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            columnValue = ["Settings"]
            rowValue = ["Ion Source", "Ion Mode", "Ionization potential", "Temperature", "MR pause", "MS settling"]
            globalVarValues = [ion_source_settings, ion_mode_settings, ionization_potential_settings, temperature_settings, mr_pause_settings, mr_settling_time_settings]
            returnValue = findColumnRowValues(table, columnValue, rowValue, globalVarValues)
            ion_source_settings = returnValue[0]
            ion_mode_settings = returnValue[1]
            ionization_potential_settings = returnValue[2]
            temperature_settings = returnValue[3]
            mr_pause_settings = returnValue[4]
            mr_settling_time_settings = returnValue[5]
        elif "Ions Monitored" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
            table_ions_monitored = table
        elif "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count4 == 0:
            globalVarValues = [analyte_1, peak_height_1, retention_time_1]
            returnValue = findAllValuesUnderColumn(table, globalVarValues)
            analyte_1 = returnValue[0]
            peak_height_1 = returnValue[1]
            retention_time_1 = returnValue[2]
            count4 += 1
        elif "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and count4 == 1:
            globalVarValues = [analyte_2, peak_height_2, retention_time_2]
            returnValue = findAllValuesUnderColumn(table, globalVarValues)
            analyte_2 = returnValue[0]
            peak_height_2 = returnValue[1]
            retention_time_2 = returnValue[2]
    nodes = []
    relationships = []
    bp_node = Node('BP', {'BP_number': bp_number})
    special_requirement_node = Node('SpecialRequirement', {'Requirements': special_requirements})
    instrumentationListNode = Node("Instrumentation",
    {"Instrumentation": bp_number, "Mass Spectrometer": g_massspec_component, "LC": bp_number,
    "Liquid Handling": g_liquidhandling_component, "Manufacturer": column_manufacturer, "LCcomponent": None})
    reagentsListNode = Node("Reagents", {"Properties": {}})
    solutionsListNode = Node("Solutions", {"Properties": {}})
    calculationParametersListNode = Node("Calculation Parameters", {"Model": regression_model})
    instrumentationNode = Node("Instrumentation", {'BP_number': bp_number})
    massSpectrometerNode = Node("MassSpectrometer", {"Name": g_massspec_component})
    LCNode = Node("LC", {"BP Number": bp_number})
    liquidHandlingNode = Node("LiquidHandling", {"Name": g_liquidhandling_component})
    manufacturerNode = Node("Manufacturer", {"Name": column_manufacturer})
    LCComponentNode = Node("LC Component", {"Name": None})
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
    compoundNode = Node("Compound", {"Analyte/L-Number": analyte_l_parent, "Form": form_parent, "Molecular_Weight": molecular_weight_parent, "Watson_ID": watson_id_parent})
    epimerNode = Node("L-number", {"Property": analyte_l_parent})
    biologicalMatrixListNode = Node("BiologicalMatrix", {
        "BP_number": bp_number,
        "Matrix": matrix,
        "Species": species,
        "Anticoagulant": anticoagulant,
        "Extraction Method": extraction_method,
        "Storage_temp": anticoagulant_temperature
    })
    manufacturerNode = Node("Manufacturer", {"Name": None})
    StandardPreparationNode = Node("BP Number", {"Property": bp_number})
    mixedIntermediateStandardSolutionNode = Node("Mixed Intermediate Standard Solution", {"Table": table_standard_solution_id})
    stockStandardSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
    workingStandardSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
    qcPreparationListNode = Node("QC Preparation", {"BP Number": bp_number})
    stockQCSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
    workingQCSolutionNode = Node("Properties", {"Dilutent": dilutent, "Storage Temp": temperature_dilutent})
    workingQCTableNode = Node("Working QC Solution ID", {"Table": table_qc_solution_id})
    matrixQCTableNode = Node("Matrix QC ID", {"Table": table_qc_id})
    matrixQCNode = Node("Matrix QC", {"Table": lloq})
    ISPreparationListNode = Node("ISPreparation", {"BP_number": bp_number})
    stockISSolutionNode = Node("StockISSolution", {
        "Name": mk_number,
        "PreparationSummary": special_requirements,
        "Use": None,
        "Storage": temperature_dilutent
    })
    workingISSolutionNode = Node("WorkingISSolution", {"Name": mk_number})
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
    return [nodes, relationships]


def populate_db(nodes: list, rels: list):
    gm.execute(operation="wipe")
    for node in nodes:
        gm.execute(operation="create_node", node=node)
    for rel in rels:
        gm.execute(operation="create_rel", rel=rel)

@app.post("/upload_pdf")
def upload_pdf(file: UploadFile = File(...)):
    filepath = f"/tmp/{file.filename}"
    filename = file.filename
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    upload_s3(filepath)
    print("parsing pdf now")
    pdf_text = parse_pdf(filename)
    searchString = pdf_text.replace("\n", " ").replace(",","")
    searchString = re.sub(' +',' ', searchString)
    tables = generate_tables(filepath)
    print("generating db content now")
    db_content = generate_DB_content(searchString, tables, filename[:7])
    print("starting db population")
    populate_db(db_content[0], db_content[1])
    os.remove(filepath)
    return {"status": "success"}