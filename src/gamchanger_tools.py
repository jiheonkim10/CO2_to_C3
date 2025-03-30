import pandas as pd
from utils.help_utils import get_unary_bulk_ids, get_f_values, get_elements

def create_gam_feature_df(co2r_data_df, ocp_df, millers): #1.5 corresponds to 0.15% FE
    
    gam_feature_df = pd.DataFrame({'composition': [],
                               'e1': [], 'e2': [], 'e3': [],
                               'f1': [],'f2': [],'f3': [],
                               'has_e1': [], 'has_e2': [], 'has_e3': [],
                               'ele1_*OCHO': [], 'ele1_*COOH': [], 'ele1_*CO': [], 'ele1_CO*COH': [], 'ele1_*CHO': [],'ele1_*C': [], 'ele1_*CO-*COOH': [], 'ele1_CO*COH-2*CO': [], 'ele1_*CHO-*CO': [], 'ele1_*C-*CHO': [], 'ele1_*C-*CO': [], 
                               'ele2_*OCHO': [], 'ele2_*COOH': [], 'ele2_*CO': [], 'ele2_CO*COH': [], 'ele2_*CHO': [],'ele2_*C': [], 'ele2_*CO-*COOH': [], 'ele2_CO*COH-2*CO': [], 'ele2_*CHO-*CO': [], 'ele2_*C-*CHO': [], 'ele2_*C-*CO': [], 
                               'ele3_*OCHO': [], 'ele3_*COOH': [], 'ele3_*CO': [], 'ele3_CO*COH': [], 'ele3_*CHO': [],'ele3_*C': [], 'ele3_*CO-*COOH': [], 'ele3_CO*COH-2*CO': [], 'ele3_*CHO-*CO': [], 'ele3_*C-*CHO': [], 'ele3_*C-*CO': [], 
                               })

    # test_compositions = ['Cr-0.50-Cu-0.50', 'Fe-0.50-Cu-0.50', 'Mn-0.50-Zn-0.50','Ni-0.50-Sn-0.50']
    compositions = co2r_data_df['composition'].unique()

    feature_data = []

    for comp in compositions:
        # print(comp)
        comp_feature_dict = {}

        unary_comps = get_unary_bulk_ids(comp, ocp_df)
        # pb_comps = get_pourbaix_phase_bulk_ids(comp, pb_df)
        # pb_comps = [item for item in pb_comps if item not in unary_comps]
        elements = get_elements(comp)
        f_values = get_f_values(comp)

        comp_feature_dict['composition'] = comp

        comp_feature_dict['e1'] = elements['e1']
        comp_feature_dict['e2'] = elements['e2']
        comp_feature_dict['e3'] = elements['e3']

        comp_feature_dict['f1'] = f_values['f1']
        comp_feature_dict['f2'] = f_values['f2']
        comp_feature_dict['f3'] = f_values['f3']

        comp_feature_dict['has_e1'] = 1 if elements['e1'] != 'None' else 0
        comp_feature_dict['has_e2'] = 1 if elements['e2'] != 'None' else 0
        comp_feature_dict['has_e3'] = 1 if elements['e3'] != 'None' else 0

        unary_count = 0
        for unary in unary_comps:
            unary_count += 1
            unary_features = ocp_df[(ocp_df['bulk_id'] == unary) & (ocp_df['slab_millers'].isin(millers))].select_dtypes(include='number').mean()

            for feature in ['*OCHO', '*COOH', '*CO', 'CO*COH', '*CHO', '*C']:
                comp_feature_dict[f'ele{unary_count}_{feature}'] = unary_features[feature]

        # pb_count = 0
        # for pb_phase in pb_comps:
        #     pb_count += 1
        #     pb_features = ocp_df[(ocp_df['bulk_id'] == pb_phase) & (ocp_df['slab_millers'].isin(millers))].select_dtypes(include='number').mean()

        #     for feature in ['*H', '*OCHO', '*COOH', '*CO', 'CO*COH', '*CHO', '*C', '*CH2']:
        #         comp_feature_dict[f'pb{pb_count}_{feature}'] = pb_features[feature]
            
        feature_data.append(comp_feature_dict)
    
    gam_feature_df = pd.concat([gam_feature_df, pd.DataFrame(feature_data)], ignore_index=True)

    phases = ['ele1', 'ele2', 'ele3']
    for element in phases:
        gam_feature_df[f'{element}_*CO-*COOH'] = gam_feature_df[f'{element}_*CO'] - gam_feature_df[f'{element}_*COOH']
        gam_feature_df[f'{element}_CO*COH-2*CO'] = gam_feature_df[f'{element}_CO*COH'] - 2 * gam_feature_df[f'{element}_*CO']
        gam_feature_df[f'{element}_*CHO-*CO'] = gam_feature_df[f'{element}_*CHO'] - gam_feature_df[f'{element}_*CO']
        gam_feature_df[f'{element}_*C-*CHO'] = gam_feature_df[f'{element}_*C'] - gam_feature_df[f'{element}_*CHO']
        gam_feature_df[f'{element}_*C-*CO'] = gam_feature_df[f'{element}_*C'] - gam_feature_df[f'{element}_*CO']

    return gam_feature_df

