import boto3
import collections
import csv
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
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
from request_classes import SearchTerm, ConfirmedValues, StatusQuery
import time


app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

s3_bucket_name = 'merckbucketnlp'


s3 = boto3.resource("s3")
textract = boto3.client('textract', region_name='us-east-1')


gm = GraphManager()

doc_status = {}
parsed_docs = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def upload_to_s3(filepath, filename):
    with open(filepath, "rb") as file:
        s3.Bucket(s3_bucket_name).put_object(Key=filename, Body=file)

def is_inside_table(line_block, table_blocks):
  for table_block in table_blocks:
    if (line_block['Geometry']['BoundingBox']['Left'] >= table_block['Geometry']['BoundingBox']['Left'] and line_block['Geometry']['BoundingBox']['Top'] >= table_block['Geometry']['BoundingBox']['Top'] and line_block['Geometry']['BoundingBox']['Left'] + line_block['Geometry']['BoundingBox']['Width'] <= table_block['Geometry']['BoundingBox']['Left'] + table_block['Geometry']['BoundingBox']['Width'] and line_block['Geometry']['BoundingBox']['Top'] + line_block['Geometry']['BoundingBox']['Height'] <= table_block['Geometry']['BoundingBox']['Top'] + table_block['Geometry']['BoundingBox']['Height']):
      return True
  return False

def extract_text(filename):
    response = textract.start_document_text_detection(
        DocumentLocation={
            'S3Object': {
                'Bucket': s3_bucket_name,
                'Name': filename
            }
        }
    )
    job_id = response['JobId']
    print("contacting AWS textract")
    while True:
        response = textract.get_document_text_detection(JobId=job_id)
        if response['JobStatus'] == 'SUCCEEDED':
            break
        elif response['JobStatus'] == 'FAILED':
            raise Exception('Textract job failed')
        print("waiting")
        time.sleep(5)
    pages = []
    response = textract.get_document_text_detection(JobId=job_id)
    pages.append(response)
    while 'NextToken' in response:
        response = textract.get_document_text_detection(JobId=job_id, NextToken=response['NextToken'])
        pages.append(response)
    text = ""
    for page in pages:
        table_blocks = [block for block in page['Blocks'] if block['BlockType'] == 'TABLE']
        for block in page['Blocks']:
            if block['BlockType'] == 'LINE' and not is_inside_table(block, table_blocks):
                text += block['Text'] + '\n'
    return text

def map_blocks(blocks, block_type):
    return {
        block['Id']: block
        for block in blocks
        if block['BlockType'] == block_type
    }

def getValue(category, column, df):
    try:
        # Find the row index where the category column contains the search term
        row_idx = df[df.iloc[:,0].str.lower().str.contains(category.lower())].index[0]
        # Find the column index where the column name contains the search term
        col_idx = df.columns.str.lower().str.contains(column.lower()).nonzero()[0][0]
        # Return the value in the corresponding cell
        return df.iloc[row_idx, col_idx]
    except (IndexError, KeyError):
        return None

def get_children_ids(block):
    for rels in block.get('Relationships', []):
        if rels['Type'] == 'CHILD':
            yield from rels['Ids']

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
    # Set up variables for pagination
    max_pages = 2
    page_count = 0
    table_list = []

    # Convert each page of the PDF file to PNG images
    images = convert_from_path(filepath)

    # Loop through each image and call Textract to analyze for tables
    for i, image in enumerate(images):
        # Check if maximum number of pages has been reached
        if page_count >= max_pages:
            break

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
        if dataframes:
            table_list += dataframes
    return table_list

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

def has_column(df, column_name):
    try:
        for col in df.columns:
            if column_name.lower() in col.lower():
                return True
        return False
    except:
        return False

def findAllValuesUnderColumn(searchTable, globalVars):
  try:
    for row in range(len(searchTable)):
      for column in range(len(searchTable.columns)):
        if column < len(globalVars):
          globalVars[column] = searchTable.iloc[row, column]
      return globalVars
  except Exception as e:
    return globalVars

def parse_extracted(searchString, tables):
    parsed = {}
    anticoagulant = ""
    matrix = ""
    species = ""
    anticoagulant = ""
    supplier = ""
    analyte_1 = ""
    peak_height_1 = ""
    retention_time_1 = ""
    analyte_2 = ""
    peak_height_2 = ""
    retention_time_2 = ""
    page1 = ""
    procedure = ""
    system_suitability = ""
    qc_sample_prep = ""
    internal_standard = ""
    # Paragraph 1
    try:
        matchpara = re.search(r'^.*?INSTRUMENTATION', searchString, re.DOTALL)

        if matchpara:
            paragraph1 = matchpara.group(0)
    except:
        pass

    # Instrumentation
    try:
        match_spec_req = re.search(r'are stored at^.*?INSTRUMENTATION', searchString, re.DOTALL)

        if match_spec_req:
            parsed["Special Requirements"] = match_spec_req.group(0)
    except:
        pass

    # Page 1
    try:
        match = re.search(r'^.*?Page 2', searchString, re.DOTALL)

        if match:
            page1 = match.group(0)
    except:
        pass

    # (QC) Sample Preparation
    try:
        match1 = re.search(r"SAMPLE PREPARATION(.*?)INTERNAL STANDARD", searchString, re.I)
        qc_sample_prep = match1.group(1)
    except:
        pass

    # Internal Standard
    try:
        match2 = re.search(r"INTERNAL STANDARD(.*?)OPERATING PARAMETERS", searchString, re.I)
        internal_standard = match2.group(1)
    except:
        pass

    # Procedure
    try:
        match3 = re.search(r"PROCEDURE(.*?)OPERATING PARAMETERS", searchString, re.I)
        procedure = match3.group(1)
    except:
        pass

    # System Suitability
    try:
        match4 = re.search(r"SYSTEM SUITABILITY(.*?)SOFTWARE AND CALCULATION", searchString, re.I)
        system_suitability = match4.group(1)
    except:
        pass
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
    prompts.append("""
    Extract the Diluent, Storage temperature, Solvent, Density, and an other condition if its used (typically in parenthesis) if used given the context below?: \n\ntext:
    """ + qc_sample_prep)
    filename = "chatgpt_output.csv"
    print("starting the chatgpt prompts")
    for prompt in prompts:
        if len(prompt) > 4080:
            prompt = prompt[:4080]
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt[:4080],
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
    keys = ["BP Number", "MK Number", "Diluent Storage Temperature", "Diluent", "Extraction Method", "Mass Spec Interface", "Ion Mode", "Drug Lower MRM",
            "Drug Upper MRM", "IS Lower MRM", "IS Upper MRM", "Calibration Range Lower", "Calibration Range Upper", "LLOQ", "Anticoagulant",
            "Matrix Storage Temperature", "Regression Model", "Reagent Troughs Provider", "Automated Workstation Tips Provider",
            "Adjustable Pipettes Provider", "Pipette Tips Manufacturer", "Microbalance Manufacturer", "Analytical Balance Manufacturer",
            "Refrigerated Centrifuge Manufacturer", "pH Meter Manufacturer", "Platesealer Manufacturer", "Column Category", "Column Manufacturer",
            "Mass Spectrometer Component", "Liquid Handling Component", "Drug L-Number", "IS L-Number", "Drug Form", "IS Form", "Drug Molecular Weight",
            "IS Molecular Weight", "Drug Watson ID", "IS Watson ID", "Matrix", "Species", "Matrix Supplier", "Standard Solution Table", "QC Solution Table",
            "QC Table", "Step Table", "Elution", "Mobile Phase A", "Mobile Phase B", "Ion Source", "Ionization Potential", "MS Temperature",
            "MR Pause", "MS Settling Time", "Ions Monitored Table", "Extract Peak Height", "Extract Retention Time", "Stock/Working Peak Height",
            "Stock/Working Retention Time", "Chromatography"]
    for key in keys:
        parsed[key] = ""
    for line in csv_content.split('\n'):
        if re.search(r'BP-\d+', line):
            try:
                parsed["BP Number"] = re.findall(r'BP-\d+', line)[0]
            except:
                pass
        elif re.search(r'MK-\d+', line):
            try:
                parsed["MK Number"] = re.findall(r'MK-\d+', line)[0]
            except:
                pass
        elif re.search(r'Chromatography Type', line, re.IGNORECASE):
            try:
                parsed["Chromatography"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'temperature', line, re.IGNORECASE):
            try:
                parsed["Diluent Storage Temperature"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'Diluent', line, re.IGNORECASE):
            try:
                parsed["Diluent"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'Extraction method', line, re.IGNORECASE):
            try:
                parsed["Extraction Method"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'Turbo ionspray', line, re.IGNORECASE):
            try:
                parsed["Mass Spec Interface"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'Ion mode', line, re.IGNORECASE):
            try:
                parsed["Ion Mode"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'MRM transition from drug', line, re.IGNORECASE):
            try:
                parsed["Drug Lower MRM"], parsed["Drug Upper MRM"] = re.findall(r'\d+\.\d+', line)
            except:
                pass
        elif re.search(r'MRM transition for internal standard', line, re.IGNORECASE):
            try:
                parsed["IS Lower MRM"], parsed["IS Upper MRM"] = re.findall(r'\d+\.\d+', line)
            except:
                pass
        elif re.search(r'Calibration Range', line, re.IGNORECASE):
            try:
                parsed["Calibration Range Lower"] = parsed["Calibration Range Upper"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'LLOQ', line, re.IGNORECASE):
            try:
                parsed["LLOQ"] = (line.split(',')[1].split()[0])
            except:
                pass
        elif re.search(r'Anticoagulant', line, re.IGNORECASE):
            try:
                parsed["Anticougulant"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'Study sample temperature', line, re.IGNORECASE):
            try:
                parsed["Matrix Storage Temperature"] = line.split(',')[1]
            except:
                pass
        elif re.search(r'\bLinear\b', line, re.IGNORECASE):
            try:
                parsed["Regression Model"] = "Linear"
            except:
                pass
    for table in tables:
        try:
            if "Category (Automation Supplies)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                try:
                    parsed["Reagent Troughs Provider"] = getValue("Reagent", str(table.columns[1]), table)
                except Exception as e:
                    pass
                try:
                    parsed["Automated Workstation Tips Provider"] = getValue("Automated", str(table.columns[1]), table)
                except Exception as e:
                    pass
        except Exception as e:
            pass

        try:
            if "Category (Pipettes)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                try:
                    parsed["Adjustable Pipettes Provider"] = getValue("Adjustable Pipettes", "Manufacturer", table)
                except Exception as e:
                    pass
                try:
                    parsed["Pipette Tips Manufacturer"] = getValue("Pipette Tips", "Manufacturer", table)
                except Exception as e:
                    pass
        except Exception as e:
            pass

        try:
            if "Category (Equipment)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                try:
                    parsed["Microbalance Manufacturer"] = getValue("Mic", "Man", table)
                except Exception as e:
                    pass
                try:
                    parsed["Analytical Balance Manufacturer"] = getValue("Ana", "Man", table) 
                except Exception as e:
                    pass      
                try:
                    parsed["Refrigerated Centrifuge Manufacturer"] = getValue("Ref", "Man", table)
                except Exception as e:
                    pass
                try:
                    parsed["pH Meter Manufacturer"] = getValue("pH Meter", "Man", table)
                except Exception as e:
                    pass
                try:
                    parsed["Platesealer Manufacturer"] = getValue("Plate", "Man", table)
                except Exception as e:
                    pass
        except Exception as e:
            pass

        try:
            if "Category (General)" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                l = len(table.to_numpy())
                parsed["Column Category"] = table.to_numpy()[l-1][0]
                parsed["Column Manufacturer"] = table.to_numpy()[l-1][1]
        except Exception as e:
            pass
        
        category_count = 0
        try:
            if "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and category_count == 0:
                try:
                    parsed["Mass Spectrometer Component"] = getValue("Mass", "Comp", table)
                except Exception as e:
                    pass
                try:
                    parsed["Liquid Handling Component"] = getValue("Liquid", "Comp", table)
                except Exception as e:
                    pass   
                category_count += 1
        except Exception as e:
            pass

        try:  
            if "Category" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and has_column(table, "Parent"):
                try:
                    parsed["Drug L-Number"] = getValue("Ana", "Par", table)
                except Exception as e:
                    pass
                try:
                    parsed["IS L-Number"] = getValue("Ana", "Int", table)
                except Exception as e:
                    pass
                try:      
                    parsed["Drug Form"] = getValue("Fo", "Par", table)
                except Exception as e:
                    pass
                try:      
                    parsed["IS Form"] = getValue("Fo", "Int", table)
                except Exception as e:
                    pass
                try:      
                    parsed["Drug Molecular Weight"] = getValue("Mol", "Par", table)
                except Exception as e:
                    pass
                try:      
                    parsed["IS Molecular Weight"] = getValue("Mol", "Int", table)
                except Exception as e:
                    pass
                try:      
                    parsed["Drug Watson ID"] = getValue("Wat", "Par", table)
                except Exception as e:
                    pass
                try:     
                    parsed["IS Watson ID"] = getValue("Wat", "Int", table)
                except Exception as e:
                    pass     
                category_count += 1 
        except Exception as e:
            pass

        try:  
            if "Species" in re.sub(' +',' ', table.columns[1].replace("\n", "")):
                globalVarValues = [matrix, species, anticoagulant, supplier]
                returnValue = findAllValuesUnderColumn(table, globalVarValues)
                parsed["Matrix"] = returnValue[0]
                parsed["Species"] = returnValue[1]
                parsed["Anticoagulant"] = returnValue[2]
                parsed["Matrix Supplier"] = returnValue[3]
        except Exception as e:
            pass

        try:
            if "Standard Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                # make a table
                parsed["Standard Solution Table"] = table
        except Exception as e:
            pass

        try:
            if "QC Solution ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                # make a table
                parsed["QC Solution Table"] = table
        except Exception as e:
            pass

        try:
            if "QC ID" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                # make a table
                parsed["QC Table"] = table
        except Exception as e:
            pass
        
        try:
            if "Step" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                # make a table
                parsed["Step Table"] = table
        except Exception as e:
            pass

        try:      
            if "LC" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and not "Profile" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                try:
                    parsed["Elution"] = getValue("Elution", "Settings", table) 
                except Exception as e:
                    pass  
                try:
                    parsed["Mobile Phase A"] = getValue("Phase A", "Settings", table) 
                except Exception as e:
                    pass  
                try:
                    parsed["Mobile Phase B"] = getValue("Phase B", "Settings", table)
                except Exception as e:
                    pass  
        except Exception as e:
            pass

        try:
            if "MS" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                try:
                    parsed["Ion Source"] = getValue("Source", "Settings", table)
                except Exception as e:
                    pass
                try:
                    parsed["Ion Mode"] = getValue("Mode", "Settings", table)
                except Exception as e:
                    pass
                try:
                    parsed["Ionization Potential"] = getValue("potential", "Settings", table)
                except Exception as e:
                    pass
                try:
                    parsed["MS Temperature"] = getValue("Temp", "Settings", table)
                except Exception as e:
                    pass
                try:
                    parsed["MR Pause"] = getValue("MR pause", "Settings", table)
                except Exception as e:
                    pass
                try:
                    parsed["MS Settling Time"] = getValue("MS set", "Settings", table)
                except Exception as e:
                    pass
        except Exception as E:
            pass

        try:
            if "Ions Monitored" in re.sub(' +',' ', table.columns[0].replace("\n", "")):
                # make a table
                parsed["Ions Monitored Table"] = table
        except Exception as e:
            pass

        analyte_count = 0

        try:
            if "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and analyte_count == 0:
                globalVarValues = [analyte_1, peak_height_1, retention_time_1]
                returnValue = findAllValuesUnderColumn(table, globalVarValues)
                parsed["Extract Peak Height"] = returnValue[1]
                parsed["Extract Retention Time"] = returnValue[2]
                analyte_count += 1
        except Exception as e:
            pass

        try:
            if "Analyte" in re.sub(' +',' ', table.columns[0].replace("\n", "")) and analyte_count == 1: 
                globalVarValues = [analyte_2, peak_height_2, retention_time_2]
                returnValue = findAllValuesUnderColumn(table, globalVarValues)
                parsed["Stock/Working Peak Height"] = returnValue[1]
                parsed["Stock/Working Retention Time"] = returnValue[2]
        except Exception as e:
            pass
    bp_match = re.search(r'(?<=BP-)\w{4}', searchString)
    if bp_match:
        parsed["BP Number"] = bp_match.group(0)
    return parsed

def populate_db(parsed: dict):
    bp = Node('BP', {'BP_number': "BP-" + parsed.get("BP Number"), 'special_requirements': parsed.get("Special Requirements")})
    lloq = Node('LLOQ', {'LLOQ': parsed.get("LLOQ")})
    uplc = Node('UPLC', {"elution": parsed.get("Elution"),
                         "mobile_phase_A": parsed.get("Mobile Phase A"),
                         "mobile_phase_B": parsed.get("Mobile Phase B"),
                         "chromatography_method": parsed.get("Chromatography")})
    ms = Node("MS", {"name": parsed.get("Mass Spectrometer Component"),
                     "ion_source": parsed.get("Ion Source"),
                     "ion_mode": parsed.get("Ion Mode"),
                     "ionization_potential": parsed.get("Ionization Potential"),
                     "temperature": parsed.get("MS Temperature"),
                     "pause": parsed.get("MR Pause"),
                     "settling_time": parsed.get("MS Settling Time"),
                     "interface": parsed.get("MS Interface")})
    sys_suit = Node("SystemSuitability", {"extract_peak_height": parsed.get("Extract Peak Height"),
                                          "extract_retention_time": parsed.get("Extract Retention Time"),
                                          "stock_peak_height": parsed.get("Stock/Working Peak Height"),
                                          "stock_retention_time": parsed.get("Stock/Working Peak Height")})
    stan_soln = Node("StandardSolution", {"diluent": parsed.get("Diluent"),
                                          "storage_temperature": parsed.get("Diluent Storage Temperature")})
    int_stan = Node("InternalStandard", {"l_number": parsed.get("IS L-Number"),
                                         "form": parsed.get("IS Form"),
                                         "molecular_weight": parsed.get("IS Molecular Weight"),
                                         "watson_id": parsed.get("IS Watson ID")})
    is_mrm = Node("MRM", {"lower": parsed.get("IS Lower MRM"),
                          "upper": parsed.get("IS Upper MRM")})
    drug = Node("Drug", {"mk_number": parsed.get("MK Number"),
                         "l_number": parsed.get("Drug L-Number"),
                         "form": parsed.get("Drug Form"),
                         "molecular_weight": parsed.get("Drug Molecular Weight"),
                         "watson_id": parsed.get("Drug Watson ID")})
    drug_mrm = Node("MRM", {"lower": parsed.get("Drug Lower MRM"),
                          "upper": parsed.get("Drug Upper MRM")})
    matrix = Node("Matrix", {"matrix": parsed.get("Matrix"),
                             "supplier": parsed.get("Matrix Supplier"),
                             "storage_temperature": parsed.get("Matrix Storage Temperature"),
                             "extraction_method": parsed.get("Extraction Method")})
    species = Node("Species", {"species": parsed.get("Species")})
    anti = Node("Anticoagulant", {"anticoagulant": parsed.get("Anticoagulant")})
    model = Node("Model", {"regression_model": parsed.get("Regression Model"),
                           "calibration_lower": parsed.get("Calibration Range Lower"),
                           "calibration_upper": parsed.get("Calibration Range Upper"),
                           "sample_size": parsed.get("Sample ")})
    nodes = [bp, lloq, uplc, ms, sys_suit, stan_soln, int_stan, is_mrm, drug, drug_mrm, matrix, species, anti, model]
    rels = []
    rels.append(Relationship(bp, lloq, 'HAS_A', {}))
    rels.append(Relationship(bp, uplc, 'HAS_A', {}))
    rels.append(Relationship(bp, ms, 'HAS_A', {}))
    rels.append(Relationship(bp, sys_suit, 'HAS_A', {}))
    rels.append(Relationship(bp, stan_soln, 'HAS_A', {}))
    rels.append(Relationship(bp, int_stan, 'HAS_A', {}))
    rels.append(Relationship(int_stan, is_mrm, 'HAS_A', {}))
    rels.append(Relationship(bp, anti, 'HAS_A', {}))
    rels.append(Relationship(bp, species, 'HAS_A', {}))
    rels.append(Relationship(species, matrix, 'HAS_A', {}))
    rels.append(Relationship(matrix, anti, 'HAS_A', {}))
    rels.append(Relationship(bp, drug, 'HAS_A', {}))
    rels.append(Relationship(drug, drug_mrm, 'HAS_A', {}))
    rels.append(Relationship(bp, model, 'HAS_A', {}))
    for node in nodes:
        gm.execute(operation="create_node", node=node)
    for rel in rels:
        gm.execute(operation="create_rel", rel=rel)

@app.get("/wipe_database")
def wipe_database():
    gm.execute(operation="wipe")
    return "success"

def handle_new_doc(filename, filepath):
    upload_to_s3(filepath, filename)
    print("beginning text extraction process")
    pdf_text = extract_text(filename)
    searchString = pdf_text.replace("\n", " ").replace(",","")
    searchString = re.sub(' +',' ', searchString)
    print("beginning table extraction process")
    tables = generate_tables(filepath)
    os.remove(filepath)
    print("parsing text and tables for BP infomration")
    parsed = parse_extracted(searchString, tables)
    doc_status[filename] = "FINISHED"
    parsed_docs[filename] = parsed
    print("finished processing " + filename)


def to_df(table: dict) -> pd.DataFrame:
    data = {}
    try:
        for col_name, content in table.items():
            data[col_name] = []
            for i in range(len(content)):
                data[col_name].append(content[str(i)])
    except:
        pass
    return pd.DataFrame(data=data)



@app.post("/save")
def save(payload: ConfirmedValues):
    values = payload.fields
    bp_number = "BP-" + values["BP Number"]
    with pd.ExcelWriter(f"/tmp/{bp_number}.xlsx") as writer:
        if "Ions Monitored Table" in values:
            to_df(values["Ions Monitored Table"]).to_excel(writer, sheet_name="Ions Monitored")
        
        if "Standard Solution Table" in values:
            to_df(values["Standard Solution Table"]).to_excel(writer, sheet_name="Standard Solution")
        
        if "QC Solution Table" in values:
            to_df(values["QC Solution Table"]).to_excel(writer, sheet_name="QC Solution")
        
        if "QC Table" in values:
            to_df(values["QC Table"]).to_excel(writer, sheet_name="QC")

        if "Step Table" in values:
            to_df(values["Step Table"]).to_excel(writer, sheet_name="Steps")
    upload_to_s3(f"/tmp/{bp_number}.xlsx", f"{bp_number}.xlsx")
    populate_db(values)


@app.post("/upload")
def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    filepath = f"/tmp/{file.filename}"
    filename = file.filename
    with open(filepath, "wb") as buffer:
        buffer.write(file.file.read())
    doc_status[filename] = "PROCESSING"
    parsed_docs[filename] = {}
    background_tasks.add_task(handle_new_doc, filename, filepath)
    return {
        "status": "SUCCESS"
    }


@app.post("/check_doc_status")
def check_doc_status(status_query: StatusQuery):
    print(doc_status)
    print(status_query.doc_name)
    des = status_query.doc_name
    status = doc_status.get(des)
    if status == None:
        status = "DNE"
    values = parsed_docs.get(des)
    return {
        "status": status,
        "values": values
    }



def fuzzy_search(search_phrases):
    search_word = re.split(r',|-|/| ', search_phrases)
    hitsDict = collections.defaultdict(int)

    session = gm.driver.session()
    for i in range(len(search_word)):
        query_string = f'''         
            CALL {{
                CALL db.index.fulltext.queryNodes("BP_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("LLOQ_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("UPLC_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("MS_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("SystemSuitability_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("StandardSolution_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("InternalStandard_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("MRM_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("Drug_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("Matrix_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("Species_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("Anticoagulant_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
                UNION ALL
                CALL db.index.fulltext.queryNodes("Model_Search", "{search_word[i]}~")
                YIELD node, score
                RETURN node, score
            }}
            WITH node
            OPTIONAL MATCH (node)-[*..2]-(n:BP)
            WHERE CASE
                WHEN "BP" IN labels(node) THEN node.BP_number IS NOT NULL
                ELSE n.BP_number IS NOT NULL
              END
            WITH *, collect(CASE WHEN "BP" IN labels(node) THEN node.BP_number ELSE n.BP_number END) as allBPNumbers
            UNWIND allBPNumbers as BPNumber
            
            MATCH (bp:BP {{BP_number: BPNumber}})
                    OPTIONAL MATCH (bp)-[*]->(m:Matrix)
                    OPTIONAL MATCH (bp)-[*]->(s:Species)
                    OPTIONAL MATCH (bp)-[*]->(d:Drug)
                    OPTIONAL MATCH (bp)-[*]->(c:UPLC)
                    OPTIONAL MATCH (bp)-[*]->(i:InternalStandard)
    
                
                
            RETURN DISTINCT apoc.convert.toJson(apoc.map.fromPairs([
            ["BP_Number: ", bp.BP_number],
            ["Matrix: ", m.matrix],
            ["Species: ", s.species],
            ["Extraction_Method: ", m.extraction_method],
            ["MK_Number: ", d.mk_number],
            ["Internal_Standard", i.watson_id],
            ["Chromatography: ", c.chromatography_method]
            ])) as json;
            '''
        listA = session.run(query_string)
        for j in listA:
            hitsDict[j] += 1
    return hitsDict

def exact_search(search_word):
    hitsDict = collections.defaultdict(int)

    session = gm.driver.session()
    query_string = f'''         
        CALL {{
            MATCH (node:BP)
            WHERE ANY(k IN ['BP_number', 'special_requirements'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:LLOQ)
            WHERE ANY(k IN ['LLOQ'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:UPLC)
            WHERE ANY(k IN ['mobile_phase_A', 'mobile_phase_B', 'chromatography_method', 'elution'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:MS)
            WHERE ANY(k IN ['ion_source', 'ion_mode', 'ionization_potential', 'temperature', 'name', 'interface', 'pause', 'settling_time'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:SystemSuitability)
            WHERE ANY(k IN ['stock_peak_height', 'stock_retention_time', 'extract_peak_height', 'extract_retention_time'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:StandardSolution)
            WHERE ANY(k IN ['storage_temperature', 'diluent'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:InternalStandard)
            WHERE ANY(k IN ['molecular_weight', 'watson_id', 'form', 'l_number'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:MRM)
            WHERE ANY(k IN ['lower', 'upper'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:Drug)
            WHERE ANY(k IN ['mk_number', 'l_number', 'form', 'molecular_weight', 'watson_id'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:Matrix)
            WHERE ANY(k IN ['matrix', 'supplier', 'storage_temperature', 'extraction_method'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:Species)
            WHERE ANY(k IN ['species'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:Anticoagulant)
            WHERE ANY(k IN ['anticoagulant'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
            UNION ALL
            MATCH (node:Model)
            WHERE ANY(k IN ['calibration_upper', 'sample_size', 'calibration_lower', 'regression_model'] WHERE toString(node[k]) =~ '(?i){search_word}.*')
            RETURN node
        }}
        WITH node
        OPTIONAL MATCH (node)-[*..2]-(n:BP)
        WHERE CASE
            WHEN "BP" IN labels(node) THEN node.BP_number IS NOT NULL
            ELSE n.BP_number IS NOT NULL
          END
        WITH *, collect(CASE WHEN "BP" IN labels(node) THEN node.BP_number ELSE n.BP_number END) as allBPNumbers
        UNWIND allBPNumbers as BPNumber
        
        MATCH (bp:BP {{BP_number: BPNumber}})
            OPTIONAL MATCH (bp)-[*]->(m:Matrix)
            OPTIONAL MATCH (bp)-[*]->(s:Species)
            OPTIONAL MATCH (bp)-[*]->(d:Drug)
            OPTIONAL MATCH (bp)-[*]->(c:UPLC)
            OPTIONAL MATCH (bp)-[*]->(i:InternalStandard)

        
        
        RETURN DISTINCT apoc.convert.toJson(apoc.map.fromPairs([
        ["BP_Number: ", bp.BP_number],
        ["Matrix: ", m.matrix],
        ["Species: ", s.species],
        ["Extraction_Method: ", m.extraction_method],
        ["MK_Number: ", d.mk_number],
        ["Internal_Standard", i.watson_id],
        ["Chromatography: ", c.chromatography_method]
        ])) as json;
        '''
    listA = session.run(query_string)
    for j in listA:
        hitsDict[j] += 1
    session.close()

    return hitsDict


@app.post("/query")
def query(query: SearchTerm):
    search_word = re.split(r',|/| ', query.term)
    hitsDict = collections.defaultdict(int)
    dictA = collections.defaultdict(int)
    dictB = collections.defaultdict(int)

    for word in search_word:
        if word == ',' or len(word) == 0:
            continue
        dictA = exact_search(word)
        if len(dictA) == 0 :
            print("do fuzzy")
            dictB = fuzzy_search(word)
            for key in dictB:
                hitsDict[key] += max(hitsDict[key], dictB[key])
        else:
            print("do exact")
            for key in dictA:
                hitsDict[key] += max(hitsDict[key], dictA[key])


    hitsDict = sorted(hitsDict.items(), key=lambda item: item[1], reverse=True)

    #print(dictA)
    # for key in hitsDict:
    #    print(str(key[0]), str(key[1]))
    return list(hitsDict)

    
