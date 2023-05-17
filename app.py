import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import FunctionTransformer
import pickle

df = pd.read_csv('test_values.csv')
# Define the function to drop columns
def drop_columns(X):
  try:
    return X.drop(['building_id', 'count_families'], axis=1)
  except:
    return X


# Define the function to get the pipeline
def get_pipeline(model):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OrdinalEncoder(), multi_class_columns), 
            ('num', Pipeline([
                ('dropper', FunctionTransformer(drop_columns)),
                ('scaler', StandardScaler())
            ]), num_columns),
        ]
    )
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline


with open('rfc_pipeline.pkl', 'rb') as f:
    rfc_pipeline = pickle.load(f)

def predict_damage_grade(building_id=4, geo_level_1_id=None, geo_level_2_id=None, geo_level_3_id=None, count_floors_pre_eq=None, age=None, 
                         area_percentage=5, height_percentage=5,land_surface_condition=None, foundation_type=None, roof_type=None, ground_floor_type=None, 
                         other_floor_type=None, position=None, plan_configuration=None, has_superstructure_adobe_mud=None, 
                         has_superstructure_mud_mortar_stone=None, has_superstructure_stone_flag=None, 
                         has_superstructure_cement_mortar_stone=None, has_superstructure_mud_mortar_brick=None, 
                         has_superstructure_cement_mortar_brick=None, has_superstructure_timber=None, 
                         has_superstructure_bamboo=None, has_superstructure_rc_non_engineered=None, 
                         has_superstructure_rc_engineered=None, has_superstructure_other=None, legal_ownership_status=None, count_families=10, has_secondary_use=None, 
                         has_secondary_use_agriculture=None, has_secondary_use_hotel=None, has_secondary_use_rental=None, 
                         has_secondary_use_institution=None, has_secondary_use_school=None, has_secondary_use_industry=None, 
                         has_secondary_use_health_post=None, has_secondary_use_gov_office=None, has_secondary_use_use_police=None, 
                         has_secondary_use_other=None, ):
                         
    inputs = {'building_id': building_id, 
              'geo_level_1_id': geo_level_1_id,
              'geo_level_2_id': geo_level_2_id,
              'geo_level_3_id': geo_level_3_id,
              'count_floors_pre_eq': count_floors_pre_eq,
              'age': age,
              'area_percentage': area_percentage,
              'height_percentage': height_percentage,
              'land_surface_condition': land_surface_condition,
              'foundation_type': foundation_type,
              'roof_type': roof_type,
              'ground_floor_type': ground_floor_type,
              'other_floor_type': other_floor_type,
              'position': position,
              'plan_configuration': plan_configuration,
              'has_superstructure_adobe_mud': int(has_superstructure_adobe_mud),
              'has_superstructure_mud_mortar_stone': int(has_superstructure_mud_mortar_stone),
              'has_superstructure_stone_flag': int(has_superstructure_stone_flag),
              'has_superstructure_cement_mortar_stone': int(has_superstructure_cement_mortar_stone),
              'has_superstructure_mud_mortar_brick': int(has_superstructure_mud_mortar_brick),
              'has_superstructure_cement_mortar_brick': int(has_superstructure_cement_mortar_brick),
              'has_superstructure_timber': int(has_superstructure_timber),
              'has_superstructure_bamboo': int(has_superstructure_bamboo),
              'has_superstructure_rc_non_engineered': int(has_superstructure_rc_non_engineered),
              'has_superstructure_rc_engineered': int(has_superstructure_rc_engineered),
              'has_superstructure_other': int(has_superstructure_other), 
              'legal_ownership_status': legal_ownership_status,
              'count_families': count_families,
              'has_secondary_use': int(has_secondary_use), 
              'has_secondary_use_agriculture': int(has_secondary_use_agriculture), 
              'has_secondary_use_hotel': int(has_secondary_use_hotel),
              'has_secondary_use_rental': int(has_secondary_use_rental), 
              'has_secondary_use_institution': int(has_secondary_use_institution),
              'has_secondary_use_school': int(has_secondary_use_school), 
              'has_secondary_use_industry':int(has_secondary_use_industry), 
              'has_secondary_use_health_post': int(has_secondary_use_health_post),
              'has_secondary_use_gov_office': int(has_secondary_use_gov_office),
              'has_secondary_use_use_police': int(has_secondary_use_use_police), 
              'has_secondary_use_other': int(has_secondary_use_other),
              
              }
    print(inputs)
    X = pd.DataFrame(inputs, index=[0])
    print("dataframe")
    y_pred = rfc_pipeline.predict(X)
    print("prediction")
    return y_pred[0]


multi_class_columns = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 
                           'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
num_columns = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'count_floors_pre_eq', 'age', 
                   'area_percentage', 'height_percentage']
binary_columns = ['has_superstructure_adobe_mud', 
                   'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 
                   'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 
                   'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 
                   'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 
                   'has_superstructure_rc_engineered', 'has_superstructure_other', 'has_secondary_use', 
                   'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 
                   'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry', 
                   'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 
                   'has_secondary_use_other']
# Define the Gradio interface
input_components = []
for col in df.columns: # Exclude the target column
    if col in multi_class_columns: # Categorical columns
        input_components.append(gr.inputs.Dropdown(df[col].unique().tolist(), label=col))
    elif col in binary_columns:
        input_components.append(gr.inputs.Radio(choices=['0', '1'], label=col))
    else: # Numerical columns
        input_components.append(gr.inputs.Number(label=col))

io = gr.Interface(predict_damage_grade, inputs=input_components, outputs="number", 
                  title="Earthquake Damage Grade Predictor", 
                  description="Predicts the damage grade (1-3) caused by an earthquake based on various features of a building.", debug=True, 
                  server_name="example", server_port=7878, share=True, allow_flagging=False,
                analytics_enabled=False, seed=123)
io.launch()