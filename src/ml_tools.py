import pandas as pd
from utils.help_utils import get_unary_bulk_ids, get_pourbaix_phase_bulk_ids, get_f_values

def filter_max_jtarget(df, target):

    result_df = pd.DataFrame()

    for value in df['composition'].unique():

        subset = df[df['composition'] == value]
        max_row = subset.loc[subset[f'j_{target}'].idxmax()]
        result_df = pd.concat([result_df, max_row.to_frame().T])

    result_df.reset_index(drop=True, inplace=True)
    return result_df

def create_pb_feature_df(co2r_data_df, ocp_df, pb_df, millers):
    
    feature_df = pd.DataFrame({'composition': [],
                               'f1': [],'f2': [],'f3': [],
                               'ele1_*OCHO': [], 'ele1_*COOH': [], 'ele1_*CO': [], 'ele1_CO*COH': [], 'ele1_*CHO': [],'ele1_*C': [], 'ele1_*CO-*COOH': [], 'ele1_CO*COH-2*CO': [], 'ele1_CO*COH-*CO-*CHO': [], 'ele1_*CHO-*CO': [], 'ele1_*C-*CHO': [], 'ele1_*C-*CO': [],
                               'ele2_*OCHO': [], 'ele2_*COOH': [], 'ele2_*CO': [], 'ele2_CO*COH': [], 'ele2_*CHO': [],'ele2_*C': [], 'ele2_*CO-*COOH': [], 'ele2_CO*COH-2*CO': [], 'ele2_CO*COH-*CO-*CHO': [], 'ele2_*CHO-*CO': [], 'ele2_*C-*CHO': [], 'ele2_*C-*CO': [],
                               'ele3_*OCHO': [], 'ele3_*COOH': [], 'ele3_*CO': [], 'ele3_CO*COH': [], 'ele3_*CHO': [],'ele3_*C': [], 'ele3_*CO-*COOH': [], 'ele3_CO*COH-2*CO': [], 'ele3_CO*COH-*CO-*CHO': [], 'ele3_*CHO-*CO': [], 'ele3_*C-*CHO': [], 'ele3_*C-*CO': [], 
                               'pb1_*OCHO': [], 'pb1_*COOH': [], 'pb1_*CO': [], 'pb1_CO*COH': [], 'pb1_*CHO': [],'pb1_*C': [], 'pb1_*CO-*COOH': [], 'pb1_CO*COH-2*CO': [], 'pb1_*CHO-*CO': [], 'pb1_*C-*CHO': [], 'pb1_*C-*CO': [], 
                               'pb2_*OCHO': [], 'pb2_*COOH': [], 'pb2_*CO': [], 'pb2_CO*COH': [], 'pb2_*CHO': [],'pb2_*C': [], 'pb2_*CO-*COOH': [], 'pb2_CO*COH-2*CO': [], 'pb2_*CHO-*CO': [], 'pb2_*C-*CHO': [], 'pb2_*C-*CO': [], })

    # test_compositions = ['Cr-0.50-Cu-0.50', 'Fe-0.50-Cu-0.50', 'Mn-0.50-Zn-0.50','Ni-0.50-Sn-0.50']
    compositions = co2r_data_df['composition'].unique()

    feature_data = []

    for comp in compositions:
        
        comp_feature_dict = {}

        unary_comps = get_unary_bulk_ids(comp, ocp_df)
        pb_comps = get_pourbaix_phase_bulk_ids(comp, pb_df)
        pb_comps = [item for item in pb_comps if item not in unary_comps]
        f_values = get_f_values(comp)

        comp_feature_dict['composition'] = comp
        comp_feature_dict['f1'] = f_values['f1']
        comp_feature_dict['f2'] = f_values['f2']
        comp_feature_dict['f3'] = f_values['f3']

        unary_count = 0
        for unary in unary_comps:
            unary_count += 1
            unary_features = ocp_df[(ocp_df['bulk_id'] == unary) & (ocp_df['slab_millers'].isin(millers))].select_dtypes(include='number').mean()

            for feature in ['*OCHO', '*COOH', '*CO', 'CO*COH', '*CHO', '*C',]:
                comp_feature_dict[f'ele{unary_count}_{feature}'] = unary_features[feature]

        pb_count = 0
        for pb_phase in pb_comps:
            pb_count += 1
            pb_features = ocp_df[(ocp_df['bulk_id'] == pb_phase) & (ocp_df['slab_millers'].isin(millers))].select_dtypes(include='number').mean()

            for feature in ['*OCHO', '*COOH', '*CO', 'CO*COH', '*CHO', '*C',]:
                comp_feature_dict[f'pb{pb_count}_{feature}'] = pb_features[feature]
            
        feature_data.append(comp_feature_dict)
    
    feature_df = pd.concat([feature_df, pd.DataFrame(feature_data)], ignore_index=True)

    phases = ['ele1', 'ele2', 'ele3', 'pb1', 'pb2']
    for element in phases:
        feature_df[f'{element}_*CO-*COOH'] = feature_df[f'{element}_*CO'] - feature_df[f'{element}_*COOH']
        feature_df[f'{element}_CO*COH-2*CO'] = feature_df[f'{element}_CO*COH'] - 2 * feature_df[f'{element}_*CO']
        feature_df[f'{element}_CO*COH-*CO-*CHO'] = feature_df[f'{element}_CO*COH'] - feature_df[f'{element}_*CO'] - feature_df[f'{element}_*CHO']
        feature_df[f'{element}_*CHO-*CO'] = feature_df[f'{element}_*CHO'] - feature_df[f'{element}_*CO']
        feature_df[f'{element}_*C-*CHO'] = feature_df[f'{element}_*C'] - feature_df[f'{element}_*CHO']
        feature_df[f'{element}_*C-*CO'] = feature_df[f'{element}_*C'] - feature_df[f'{element}_*CO']

    return feature_df