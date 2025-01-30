# import dependencies 
import json
import transformers
import torch
import os

from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

# Notify
print("Successfull import")

# ------------------------------------------
# Load 02 gpu if available. 02 cpu otherwise
# ------------------------------------------
def load_gpus():

    # Vérifier la disponibilité des GPU
    device_0 = 0 if torch.cuda.device_count() > 0 else -1
    device_1 = 1 if torch.cuda.device_count() > 1 else -1

    # Initialiser deux pipelines sur les deux GPU
    classifier_gpu_0 = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device_0)
    classifier_gpu_1 = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=device_1)

    # Notify
    print("Classifiers loaded successfully")
    
    return classifier_gpu_0, classifier_gpu_1

# ------------------------------------------
# Create directory if not exist.
# ------------------------------------------
def create_directory_if_not_exists(directory_path):
    """
    Crée un répertoire s'il n'existe pas.

    :param directory_path: Chemin complet du répertoire à créer.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Répertoire créé : {directory_path}")
    else:
        print(f"Le répertoire existe déjà : {directory_path}")
        

# ------------------------------------------
# Load json file content
# ------------------------------------------
def load_json(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"{file_path} loaded successfully")
    return data

# ------------------------------------------
# Get category code from categorie name
# ------------------------------------------
def get_category_code_from_name(name, category_list):
    for category in category_list:
        if category['Name'] == name:
            return category['Code']
    return None
    
# ------------------------------------------
# process batch for level 1 prediction
# ------------------------------------------
def process_batch_level_1(batch, candidate_subjects, classifier):
    filenames_batch = [record['filename'] for record in batch]
    input_text_batch = [record['text'] for record in batch]
    paths_batch = [record['path'] for record in batch]

    # Obtenir les prédictions
    output_batch = [
        classifier(input_text, candidate_subjects, multi_label=True)
        for input_text in input_text_batch
    ]

    # Formater les résultats
    results = []
    for j, output in enumerate(output_batch):
        results.append({
            "filename": filenames_batch[j],
            "text": input_text_batch[j],
            "path": paths_batch[j],
            "predictions": [
                {'code': get_category_code_from_name(label), 'label': label, 'score': score}
                for label, score in zip(output['labels'][:2], output['scores'][:2])
            ]
        })
    return results

# ------------------------------------------
# level 1 predictions function
# ------------------------------------------
def level_1_prediction(
    prediction_path, 
    start_index, 
    end_index, 
    batch_size, 
    tibkat_record_list, 
    level_1_candidate_labels, 
    classifier_gpu_0, 
    classifier_gpu_1,
    ):
    
    level_1_output_list = []

    start_index = 4000
    end_index = None

    test_list = tibkat_record_list[start_index:]

    if end_index is not None:
        test_list = tibkat_record_list[start_index:end_index]
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(test_list), batch_size * 2):  # Charger 2x batch_size pour les deux GPU
            
            # Diviser en sous-lots pour chaque GPU
            batch_gpu_0 = test_list[i:i + batch_size]
            batch_gpu_1 = test_list[i + batch_size:i + batch_size * 2]

            # Soumettre les tâches au pool de threads
            future_0 = executor.submit(process_batch_level_1, batch_gpu_0, level_1_candidate_labels, classifier_gpu_0) if batch_gpu_0 else None
            future_1 = executor.submit(process_batch_level_1, batch_gpu_1, level_1_candidate_labels, classifier_gpu_1) if batch_gpu_1 else None

            # Récupérer les résultats
            results_gpu_0 = future_0.result() if future_0 else []
            results_gpu_1 = future_1.result() if future_1 else []

            # Consolider les résultats
            level_1_output_list.extend(results_gpu_0)
            level_1_output_list.extend(results_gpu_1)

            # Sauvegarde des checkpoints tous les 500 prédictions
            if len(level_1_output_list) % 500 == 0:
                level_1_prediction_checkpoint = f"{prediction_path}/predictions_{start_index}_level_1_checkpoint_{len(level_1_output_list)}.json"
                with open(level_1_prediction_checkpoint, "w", encoding="utf-8") as f:
                    json.dump(level_1_output_list, f, indent=4, ensure_ascii=False)
                print(f"Checkpoint {len(level_1_output_list)} enregistré")

            print(f"Traitement {len(level_1_output_list)}/{len(test_list)} terminé")

    # save
    level_1_prediction_file = f"{prediction_path}/predictions_level_1_final.json"
    with open(level_1_prediction_file, "w", encoding="utf-8") as f:
        json.dump(level_1_output_list, f, indent=4, ensure_ascii=False)

    # Notify
    print("Level 1 prediction finished")
    
    return level_1_output_list

# ------------------------------------------
# level 2 predictions function
# ------------------------------------------
def level_2_prediction(prediction_path, gnd_data, start_index, end_index, batch_size, classifier_gpu_0, classifier_gpu_1):
    level_1_output_list_file = f"{prediction_path}/predictions_level_1_final.json"

    level_1_output_list = []
    
    # load level 1 prediction
    level_1_output_list = load_json(level_1_output_list_file)

    # %% [code] {"jupyter":{"outputs_hidden":false}}
    def get_gnd_subjects_from_categorie(categorie_code):

        # output list
        gnd_subjects = []

        for item in gnd_data:
            # check if it is subject with exactly 'categorie_code'
            if item['Classification Number'] == categorie_code:
                gnd_subjects.append(item)

        return gnd_subjects

    # Notify
    print("Function get_gnd_subjects_from_sub_parent_class is ready")

    # Input list for level 2 predictions
    tibkat_record_file_2 = []

    for item in level_1_output_list:
        candidate_subject = []
        for categorie in item['predictions'][:2]:
            candidate_subject.extend(get_gnd_subjects_from_categorie(categorie['code']))

        # add item
        tibkat_record_file_2.append({
            "filename": item['filename'],
            "text": item['text'],
            "path": item['path'],
            "candidate_subjects": candidate_subject
        })

    # Notify
    print("Input list for level 2 prediction is ready")

    # %% [code] {"jupyter":{"outputs_hidden":false}}
    def get_gnd_code_from_name(gnd_name, filename):
        # get candidate subject of filename
        for record in tibkat_record_file_2:
            if record['filename'] == filename:
                candidate_subjects = record['candidate_subjects']

                for subject in candidate_subjects:
                    if subject['Name'] == gnd_name:
                        return subject['Code']

    # Notify
    print("get_gnd_code_from_name function is defined successfully")

    # Extract `Name` of each item in `tibkat_record_file_2`
    level_2_candidate_subjects = [
        [label["Name"] for label in record['candidate_subjects'] if "Name" in label]
        for record in tibkat_record_file_2
    ]

    level_2_top_subject = 50

    # Notify
    print("Level 2 candidates subjects are ready")

    # Liste vide pour stocker les résultats
    level_2_output_list = []

    test_list_2 = tibkat_record_file_2[start_index:]

    if end_index is not None:
        test_list_2 = tibkat_record_file_2[start_index:end_index]

    # Fonction pour traiter un lot pour le niveau 1
    def process_batch_2(batch, candidate_subjects_batch, classifier):
        filenames_batch = [record['filename'] for record in batch]
        input_text_batch = [record['text'] for record in batch]
        paths_batch = [record['path'] for record in batch]

        # Obtenir les prédictions
        output_batch = [
            classifier(input_text, candidate_subjects, multi_label=True)
            for input_text, candidate_subjects in zip(input_text_batch, candidate_subjects_batch)
        ]

        # Formater les résultats
        results = []
        for j, output in enumerate(output_batch):
            results.append({
                "filename": filenames_batch[j],
                "text": input_text_batch[j],
                "path": paths_batch[j],
                "predictions": [
                    {'code': get_gnd_code_from_name(label, filenames_batch[j]), 'label': label, 'score': score}
                    for label, score in zip(output['labels'][:level_2_top_subject], output['scores'][:level_2_top_subject])
                ]
            })
        return results

    # Traiter les lots en parallèle
    with ThreadPoolExecutor(max_workers=2) as executor:
        for i in range(0, len(test_list_2), batch_size * 2):  # Charger 2x batch_size pour les deux GPU
            # Diviser en sous-lots pour chaque GPU
            batch_gpu_0 = test_list_2[i:i + batch_size]
            batch_gpu_1 = test_list_2[i + batch_size:i + batch_size * 2]

            candidate_subjects_gpu_0 = level_2_candidate_subjects[i:i + batch_size]
            candidate_subjects_gpu_1 = level_2_candidate_subjects[i + batch_size:i + batch_size * 2]

            # Soumettre les tâches au pool de threads
            future_0 = executor.submit(process_batch_2, batch_gpu_0, candidate_subjects_gpu_0, classifier_gpu_0) if batch_gpu_0 else None
            future_1 = executor.submit(process_batch_2, batch_gpu_1, candidate_subjects_gpu_1, classifier_gpu_1) if batch_gpu_1 else None

            # Récupérer les résultats
            results_gpu_0 = future_0.result() if future_0 else []
            results_gpu_1 = future_1.result() if future_1 else []

            # Consolider les résultats
            level_2_output_list.extend(results_gpu_0)
            level_2_output_list.extend(results_gpu_1)

            # Sauvegarde des checkpoints tous les 500 prédictions
            if len(level_2_output_list) % 500 == 0:
                level_2_prediction_checkpoint = f"{prediction_path}/predictions_{start_index}_level_2_checkpoint_{len(level_2_output_list)}.json"
                with open(level_2_prediction_checkpoint, "w", encoding="utf-8") as f:
                    json.dump(level_2_output_list, f, indent=4, ensure_ascii=False)
                print(f"Checkpoint {len(level_2_output_list)} enregistré")

            print(f"Traitement {len(level_2_output_list)}/{len(test_list_2)} terminé")

    # save
    level_2_prediction_file = f"{prediction_path}/predictions_level_2_final.json"
    with open(level_2_prediction_file, "w", encoding="utf-8") as f:
        json.dump(level_2_output_list, f, indent=4, ensure_ascii=False)

    # Notify
    print("Level 2 prediction finished")
    
    return level_2_output_list
      
def main():
     
    # variables --------------------------------------------------
    
    # prediction level
    prediction_level = 2
    
    # dataset split : tib-core | all
    split = "tib-core"
    
    # prediction path
    prediction_path = f"/Results/{split}"
    
    # Load categories
    categories_file_path = "gnd-classification-for-gnd-subjects-filter.json" 
    
    # Load GND subject
    gnd_taxonomy_path = "GND-Subjects-all.json"
    
    # Tibkat record file path
    tibkat_record_file_path = f"tibkat_test_{split}_subjects.json"

    
    # functions calls --------------------------------------------------
    
    # load classifiers
    classifier_gpu_0, classifier_gpu_1 = load_gpus()
    
    # create perdiction path
    # create_directory_if_not_exists(prediction_path)
    
    if(prediction_level == 1):
    
        # load categories
        categories = load_json(categories_file_path)
        
        # extract categories names
        level_1_candidate_labels = [item["Name"] for item in categories if "Name" in item]
        
        # load record
        tibkat_record_list = load_json(tibkat_record_file_path)
        
        print(
            """
            ---------------------------------------------------------------------------
            Start level 1 prediction
            ---------------------------------------------------------------------------
            """
            )
        
        # get level 1 output list
        level_1_output_list = level_1_prediction(
            prediction_path=prediction_path,
            start_index=0,
            end_index=None,
            batch_size=4,
            tibkat_record_list=tibkat_record_list,
            level_1_candidate_labels=level_1_candidate_labels,
            classifier_gpu_0=classifier_gpu_0,
            classifier_gpu_1=classifier_gpu_1,
        )
        
        prediction_level = 2
    
    
    if(prediction_level == 2):
        
        # load gnd taxonomy
        gnd_data = load_json(gnd_taxonomy_path)
        
        print(
            """
            ---------------------------------------------------------------------------
            Start level 2 prediction
            ---------------------------------------------------------------------------
            """
            )
        
        # get level 2 prediction
        level_2_output_list = level_2_prediction(
            prediction_path=prediction_path,
            gnd_data=gnd_data,
            start_index=0,
            end_index=None,
            batch_size=4,
            classifier_gpu_0=classifier_gpu_0,
            classifier_gpu_1=classifier_gpu_1,
        )
        
# call main functin
main()
