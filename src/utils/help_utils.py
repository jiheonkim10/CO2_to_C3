import os
import pandas as pd
import ast

### Data Import

def convert_to_python_object(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value

def get_data_df(directory_path, file_name, columns_for_None = None):
    
    directory_path = os.path.expanduser(directory_path)
    file_path = os.path.join(directory_path, file_name)

    if file_name.endswith('.csv'):
        file_df = pd.read_csv(file_path)

    elif file_name.endswith('.xlsx'):
        file_df = pd.read_excel(file_path)

    else:
        'Invalid file format'
    
    if columns_for_None is None:
        file_df[file_df.columns] = file_df[file_df.columns].map(convert_to_python_object)
    else:
        file_df[columns_for_None] = file_df[columns_for_None].map(lambda x: None if pd.isna(x) else x)
    
    return file_df

# Get functions

def get_unary_bulk_ids(composition, ocp_df):
    elements = composition.split('-')
    unary_phases = []
    unary_bulk_ids = []

    for i in range(0, len(elements), 2):
        elem = elements[i]
        unary_phases.append(f"{elem}-1.000")

    for unary in unary_phases:
        bulk_id = ocp_df[ocp_df['slab_comp'] == unary]['bulk_id'].values[0]
        unary_bulk_ids.append(bulk_id)
        
    return unary_bulk_ids

def get_pourbaix_phase_bulk_ids(composition, pb_df):
    pourbaix_phase_bulk_ids = pb_df[pb_df['composition'] == composition]['pourbaix_phases'].values[0]
    return pourbaix_phase_bulk_ids

def get_elements(composition):
    elements = composition.split('-')

    element_dict = {}

    for i in range(0, len(elements), 2):
        element_dict[f"e{i // 2 + 1}"] = elements[i]
    
    for key in ['e1', 'e2', 'e3']:
        if key not in element_dict:
            element_dict[key] = 'None'

    return element_dict

def get_f_values(composition):
    elements = composition.split('-')
    f_values = {}

    for i in range(1, len(elements), 2):
        f_values[f"f{i // 2 + 1}"] = float(elements[i])

    for key in ['f1', 'f2', 'f3']:
        if key not in f_values:
            f_values[key] = 0.0

    return f_values